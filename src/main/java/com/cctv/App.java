package com.cctv;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameGrabber.Exception;
import org.bytedeco.javacv.FFmpegLogCallback;
import org.bytedeco.javacv.Frame;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_cudaobjdetect.CudaCascadeClassifier;
import org.bytedeco.opencv.opencv_face.*;

import org.bytedeco.opencv.opencv_objdetect.*;


/**
 * Hello world!
 *
 */

public class App 
{
    public static void main( String[] args ) throws Exception
    {   
    	
    	train traindata = new train();
        traindata.trainingimgfile("/home/chan/train", "/home/chan/classify/dasw");
 	   // traindata.faceconverttojpg("/home/chan/train", "http://192.168.1.119:9083/video");
        FFmpegLogCallback.set();
    	FFmpegFrameGrabber cam = new FFmpegFrameGrabber("http://192.168.1.119:9083/video");
    	CanvasFrame canvas = new CanvasFrame("Some Title", CanvasFrame.getDefaultGamma()/cam.getGamma());
    	CascadeClassifier facecas = new CascadeClassifier("/home/chan/my-cctv/data/lbpcascade_frontalface_improved.xml");
        FaceRecognizer lbphFaceRecognizer = LBPHFaceRecognizer.create();
        lbphFaceRecognizer.read("/home/chan/model/model1.YAML");
    	canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
    	 
        cam.start();
        while(true) {
        	//captured_frame= converter.convert(cam.grab());
        	if (cam.grab() == null) {
                System.out.println("!!! Failed cvQueryFrame");
                return;
            }
        	OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
        	Mat source = new Mat();
        	source = converter1.convert(cam.grab());
        	Mat grayscale = new Mat();
            //opencv_imgproc.cvtcolor(source,destination,type);
        	opencv_imgproc.cvtColor(source, grayscale, opencv_imgproc.COLOR_RGB2GRAY);
        	opencv_imgproc.equalizeHist(grayscale, grayscale);
        	Point p = new Point();
        	RectVector faces = new RectVector();
            facecas.detectMultiScale(grayscale, faces);
            for(int i = 0; i<faces.size();i++) {
            	Rect r = faces.get(i);
            	Mat face = new Mat(grayscale, r);
            	IntPointer label = new IntPointer(1);
                DoublePointer confidence = new DoublePointer(1);
                lbphFaceRecognizer.predict(face, label, confidence);
                int prediction = label.get(0);
                opencv_imgproc.rectangle(source, r, new Scalar(0, 255, 0, 1));
                File folder = new File ("/home/chan/train");
                File[] listoffiles = folder.listFiles();
                String name = "";
                String c = Integer.toString(prediction);
                System.out.println(c);
                for(int l= 0; l<listoffiles.length;l++) {
                	if(listoffiles[l].getName().contains(c)){
                		System.out.println(listoffiles[l].getName());
                	 name = listoffiles[l].getName().replaceAll(".jpg", "");
                	 name = name.replaceAll("[^a-zA-Z0-9]", " ");  
                	 name = name.replaceAll("[0-9]", "");
                		
                	}
                	
                }
                
                String box_text = "Prediction = " + prediction+" "+name;
                int pos_x = Math.max(r.tl().x() - 10, 0);
                int pos_y = Math.max(r.tl().y() - 10, 0);
                
                 if(c==null) {
                	 
                	box_text = "Prediction =  unknown user";
                }
                 opencv_imgproc.putText(source, box_text, new Point(pos_x, pos_y),opencv_imgproc.FONT_HERSHEY_PLAIN, 1.0, new Scalar(0, 255, 0, 2.0));
            }
            

            Frame frame = converter1.convert(source);
        	cam.setFrameRate(60);
        	canvas.showImage(frame); 
        }   
    }
}
