import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Garba Pose Detection',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: GarbaScreen(camera: camera),
    );
  }
}

class GarbaScreen extends StatefulWidget {
  final CameraDescription camera;
  const GarbaScreen({required this.camera});

  @override
  State<GarbaScreen> createState() => _GarbaScreenState();
}

class _GarbaScreenState extends State<GarbaScreen> {
  late CameraController _controller;
  late PoseDetector _poseDetector;
  String _predictedStep = 'Waiting...';
  int _score = 0;
  List<String> _feedback = [];
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializePoseDetector();
  }

  void _initializeCamera() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
    );

    await _controller.initialize();
    _startPoseDetection();
    setState(() {});
  }

  void _initializePoseDetector() {
    _poseDetector = PoseDetector(
      options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
    );
  }

  void _startPoseDetection() async {
    if (!_controller.value.isInitialized) return;

    _controller.startImageStream((CameraImage image) async {
      if (_isProcessing) return;
      _isProcessing = true;

      try {
        final inputImage = InputImage.fromBytes(
          bytes: image.planes[0].bytes,
          metadata: InputImageMetadata(
            size: Size(image.width.toDouble(), image.height.toDouble()),
            rotation: InputImageRotation.rotation0deg,
            format: InputImageFormat.nv21,
            bytesPerRow: image.planes[0].bytesPerRow,
          ),
        );

        final poses = await _poseDetector.processImage(inputImage);

        if (poses.isNotEmpty) {
          final pose = poses[0];
          List<double> landmarks = [];

          for (var landmark in pose.landmarks) {
            landmarks.addAll([
              landmark.position.dx,
              landmark.position.dy,
              landmark.z ?? 0,
              landmark.inFrameLikelihood,
            ]);
          }

          _sendToServer(landmarks);
        }
      } catch (e) {
        print('Error: $e');
      }

      _isProcessing = false;
    });
  }

  void _sendToServer(List<double> landmarks) async {
    try {
      final response = await http.post(
        Uri.parse('http://10.21.100.105:8000/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'landmarks': landmarks}),
      ).timeout(Duration(seconds: 2));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _predictedStep = data['step'] ?? 'Unknown';
          _score = data['score'] ?? 0;
          _feedback = List<String>.from(data['feedback'] ?? []);
        });
      }
    } catch (e) {
      print('Server error: $e');
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _poseDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text('Garba Pose Detection')),
      body: Stack(
        children: [
          CameraPreview(_controller),
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Step: ${_predictedStep}',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Score: $_score/100',
                    style: TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 10),
                  ..._feedback.map((f) => Text(
                    f,
                    style: TextStyle(color: Colors.orangeAccent),
                  )),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}