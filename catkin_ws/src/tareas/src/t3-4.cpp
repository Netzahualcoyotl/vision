#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "tareas/mensaje.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
using namespace cv;
using namespace std;

void funcion2(const tareas::mensaje::ConstPtr& msg){ cout << "Nombre de la imagen en el servidor" << msg->nimagen << endl;
}

int main(int argc, char** argv)
{
    vector<Mat> images;
    vector<String> img_files, yaml_files;
    string pattern(argv[1]);   
    ros::init(argc, argv, "Imagen_publicada");
    ros::NodeHandle nodoEmisor, nodoReceptor;
    image_transport::ImageTransport it(nodoEmisor);
    image_transport::Publisher pub = it.advertise("ImagenAConsultar", 1);
    ros::Subscriber subscriptor = nodoReceptor.subscribe("Nombre", 0, funcion2);
    int i = 0;
    ROS_INFO("nodo_receptor creado y registrado");       
    glob("dataset/" + pattern + "/*.jpg", img_files, false);
    for(size_t i = 0; i < img_files.size(); i++)
        images.push_back(imread(img_files[i]));
                
    cout << "Consultando centroides para: " << pattern << endl;
    cout << "Archivos a consultar: " << img_files.size() << endl;         
    ros::Duration segundosadormir(2);    
    while(nodoEmisor.ok()){   
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", images[i]).toImageMsg();        
        pub.publish(msg); 
        ros::spinOnce();
        segundosadormir.sleep(); 
        cout << "i: " << i << endl;
        if (i == images.size()-1) i = 0;
        i++;    
     }   
    return 0;
}
