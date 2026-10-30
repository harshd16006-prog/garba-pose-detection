import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:http/http.dart' as http;

// ====== SETTINGS (yahan badlo) ======
const String kServerUrl = 'https://garba-pose-detection.onrender.com/predict'; // apna laptop ka IPv4
const int kSendEveryMs = 400; // itni der baad server ko bhejna
const bool kFlipLikeTraining = true; // training mein webcam image mirror thi
// ====================================

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final camera = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras.first,
    );
    return MaterialApp(
      title: 'Garba Pose Detection',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: GarbaScreen(camera: camera),
    );
  }
}

class GarbaScreen extends StatefulWidget {
  final CameraDescription camera;
  const GarbaScreen({super.key, required this.camera});

  @override
  State<GarbaScreen> createState() => _GarbaScreenState();
}

class _GarbaScreenState extends State<GarbaScreen> {
  CameraController? _controller;
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
  );

  String _predictedStep = 'WAITING...';
  int _score = 0;
  List<String> _feedback = [];
  String _status = 'Camera start ho raha hai...';
  bool _isProcessing = false;
  int _frames = 0;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);

  static const Map<DeviceOrientation, int> _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup:
          Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );
    try {
      await controller.initialize();
    } catch (e) {
      if (mounted) setState(() => _status = 'Camera error: $e');
      return;
    }
    if (!mounted) return;
    _controller = controller;
    await controller.startImageStream(_processImage);
    setState(() => _status = 'Pose dikhao (poora body frame mein)');
  }

  InputImageRotation? _rotation() {
    final controller = _controller;
    if (controller == null) return null;
    final sensorOrientation = widget.camera.sensorOrientation;
    if (Platform.isIOS) {
      return InputImageRotationValue.fromRawValue(sensorOrientation);
    }
    var comp = _orientations[controller.value.deviceOrientation];
    if (comp == null) return null;
    if (widget.camera.lensDirection == CameraLensDirection.front) {
      comp = (sensorOrientation + comp) % 360;
    } else {
      comp = (sensorOrientation - comp + 360) % 360;
    }
    return InputImageRotationValue.fromRawValue(comp);
  }

  // Left <-> Right landmark ka jodi (training mein image mirror thi)
  int _mirrorIndex(int i) {
    if (i == 0) return 0;
    if (i <= 3) return i + 3;
    if (i <= 6) return i - 3;
    return i.isOdd ? i + 1 : i - 1;
  }

  // YUV420 (3 planes) -> NV21 (Y plane + interleaved V,U)
  Uint8List _yuv420ToNv21(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final out = Uint8List(width * height * 3 ~/ 2);
    var idx = 0;
    for (var row = 0; row < height; row++) {
      final start = row * yPlane.bytesPerRow;
      out.setRange(idx, idx + width, yPlane.bytes, start);
      idx += width;
    }
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;
    final vRowStride = vPlane.bytesPerRow;
    final vPixelStride = vPlane.bytesPerPixel ?? 1;
    for (var row = 0; row < height ~/ 2; row++) {
      for (var col = 0; col < width ~/ 2; col++) {
        out[idx++] = vPlane.bytes[row * vRowStride + col * vPixelStride];
        out[idx++] = uPlane.bytes[row * uvRowStride + col * uvPixelStride];
      }
    }
    return out;
  }

  Future<void> _processImage(CameraImage image) async {
    if (_isProcessing) return;
    if (DateTime.now().difference(_lastSent).inMilliseconds < kSendEveryMs) {
      return;
    }
    _isProcessing = true;
    _frames++;
    if (mounted) setState(() => _status = 'frames=$_frames (frame aa rahe hain)');
    try {
      final rotation = _rotation();
      if (rotation == null || image.planes.isEmpty) {
        if (mounted) {
          setState(() => _status =
              'DEBUG: rot=$rotation planes=${image.planes.length}');
        }
        return;
      }

      // Camera 3 planes (YUV420) deta hai to NV21 mein badlo
      final Uint8List bytes = image.planes.length == 1
          ? image.planes.first.bytes
          : _yuv420ToNv21(image);

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: InputImageFormat.nv21,
          bytesPerRow: image.width,
        ),
      );

      final poses = await _poseDetector
          .processImage(inputImage)
          .timeout(const Duration(seconds: 4));
      if (poses.isEmpty) {
        if (mounted) setState(() => _status = 'Pose nahi dikha. Thoda peeche jao');
        return;
      }

      // Rotation ke baad image ki chaudai/lambai
      final rotated = rotation == InputImageRotation.rotation90deg ||
          rotation == InputImageRotation.rotation270deg;
      final w = (rotated ? image.height : image.width).toDouble();
      final h = (rotated ? image.width : image.height).toDouble();

      final pose = poses.first;
      final points = List<List<double>>.generate(33, (i) {
        final lm = pose.landmarks[PoseLandmarkType.values[i]];
        if (lm == null) return [0.0, 0.0, 0.0, 0.0];
        return [lm.x / w, lm.y / h, lm.z / w, lm.likelihood];
      });

      final flat = <double>[];
      for (var i = 0; i < 33; i++) {
        final p = kFlipLikeTraining ? points[_mirrorIndex(i)] : points[i];
        final x = kFlipLikeTraining ? 1.0 - p[0] : p[0];
        flat.addAll([x, p[1], p[2], p[3]]);
      }

      _lastSent = DateTime.now();
      await _sendToServer(flat);
    } catch (e) {
      debugPrint('Error: $e');
      if (mounted) setState(() => _status = 'DEBUG error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Future<void> _sendToServer(List<double> landmarks) async {
    try {
      final response = await http
          .post(
            Uri.parse(kServerUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'landmarks': landmarks}),
          )
          .timeout(const Duration(seconds: 3));

      if (!mounted) return;
      if (response.statusCode != 200) {
        setState(() => _status = 'Server error ${response.statusCode}');
        return;
      }
      final data = jsonDecode(response.body);
      if (data['success'] == true) {
        setState(() {
          _predictedStep = (data['step'] ?? 'Unknown').toString().toUpperCase();
          _score = (data['score'] as num?)?.round() ?? 0;
          _feedback = List<String>.from(data['feedback'] ?? []);
          _status = '';
        });
      } else {
        setState(() => _status = 'Server: ${data['error']}');
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Server nahi mil raha. IP aur WiFi check karo');
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _poseDetector.close();
    super.dispose();
  }

  Color _scoreColor() {
    if (_score >= 80) return Colors.greenAccent;
    if (_score >= 50) return Colors.amberAccent;
    return Colors.redAccent;
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(_status),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Garba Pose Detection v4')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller),
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Step: $_predictedStep',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'Score: $_score/100',
                    style: TextStyle(
                      color: _scoreColor(),
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 10),
                  ..._feedback.map(
                    (f) => Text(f, style: const TextStyle(color: Colors.orangeAccent)),
                  ),
                  if (_status.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Text(_status, style: const TextStyle(color: Colors.white70)),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
