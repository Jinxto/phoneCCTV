package com.cctv;


import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.Scanner;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameGrabber.Exception;
import org.bytedeco.javacv.FFmpegLogCallback;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;


public class train{
	String traindir;
	String classifydir;
	String link;
	Mat source;
	Mat gray;
	//FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
    // FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
     FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
	public train() {
		
	}
	void trainingimgfile(String traindir,String classifydir) {
		this.classifydir=classifydir;
		this.traindir=traindir;
		Mat testimage = imread(classifydir,IMREAD_GRAYSCALE);
	     File root = new File(traindir);
	     FilenameFilter imgFilter = new FilenameFilter() {
	    	 public boolean accept(File dir, String name) {
	                name = name.toLowerCase();
	                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
	     }
	
	};
	 File[] imageFiles = root.listFiles(imgFilter);
     MatVector images = new MatVector(imageFiles.length);
     Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
     IntBuffer labelsBuf = labels.createBuffer();
     int counter = 0;
     for (File image : imageFiles) {
         Mat img = imread(image.getAbsolutePath(), IMREAD_GRAYSCALE);
         int label = Integer.parseInt(image.getName().split("\\-")[0]);
         images.put(counter, img);
         labelsBuf.put(counter, label);
         counter++;
     }
   

     faceRecognizer.train(images, labels);

     IntPointer label = new IntPointer(1);
     DoublePointer confidence = new DoublePointer(1);
     faceRecognizer.predict(testimage, label, confidence);
     int predictedLabel = label.get(0);
     //faceRecognizer.write("/home/chan/model/model1.YAML");
     System.out.println("Predicted label: " + predictedLabel);
	
 }
	void faceconverttojpg(String traindir, String link) throws Exception {
		this.link = link;
		this.traindir = traindir;
		FFmpegFrameGrabber cam = new FFmpegFrameGrabber(link);
		FFmpegLogCallback.set();
    	CascadeClassifier facecas = new CascadeClassifier("/home/chan/my-cctv/data/lbpcascade_frontalface_improved.xml");
		cam.start();
        while(true) {
        	if(cam.grab()==null) {
        		System.out.println("Camera is offline");
        		return;
        	}
        	OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
    		Mat testimage = converter1.convert(cam.grab());
    		source = testimage;
    		Mat grayscale = new Mat();
    		opencv_imgproc.cvtColor(source, grayscale, opencv_imgproc.COLOR_RGB2GRAY);
        	opencv_imgproc.equalizeHist(grayscale, grayscale);
        	//your face area
        	RectVector faces = new RectVector();
            facecas.detectMultiScale(grayscale, faces);
            if(faces.empty()) {
            	System.out.println("face not detected");
            	//change frame to image code here
            }
            if(!faces.empty()) {
            	System.out.print("face detected");
            }
            	
            
         
        }
		
	}
}
