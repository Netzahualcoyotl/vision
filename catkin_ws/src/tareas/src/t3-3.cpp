#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "tareas/mensaje.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#define N 1
using namespace cv;
using namespace std;

vector<Vec3f> disturb_centroids(vector<Vec3f> centroids, float epsilon)
{
vector<Vec3f> new_centroids;
Vec3f v;
    for(int i = 0; i < centroids.size(); i++){
        v(0) = ((float) rand()) / (float) RAND_MAX; 
        v(1) = ((float) rand()) / (float) RAND_MAX;
        v(2) = ((float) rand()) / (float) RAND_MAX;
        v = epsilon * v / norm(v);
        new_centroids.push_back(centroids[i] + v);
        new_centroids.push_back(centroids[i] - v);
        }

return new_centroids;

}

vector<Vec3f> get_centroids(vector<Vec3b> dat, float m, float epsilon, float tol)
{
vector<Vec3f> centroids, e((0,0,0)),data(dat.size(),(0,0,0));
Scalar b = mean(dat);
Vec3f p(b(0),b(1),b(2)),c;
centroids.push_back(p);

for(int i = 0; i < dat.size(); i++) {
    data[i](0) = dat[i](0);
    data[i](1) = dat[i](1);
    data[i](2) = dat[i](2);
    }

float delta = tol + 1;
int minindex=0;


 while (centroids.size() < m)
     {
         centroids = disturb_centroids(centroids, epsilon);
         
         delta = tol + 1;
         while (delta > tol)
             {
                 vector<vector<Vec3f> > clusters(centroids.size(), e);
                 vector<float> DD(centroids.size(),0);
                 vector<Vec3f> new_centroids(centroids.size(),(0,0,0));
                 for (int i = 0;i < data.size(); i++){
                     for (int j= 0; j < centroids.size(); j++) {
                         c(0) = data[i](0)-centroids[j](0);
                         c(1) = data[i](1)-centroids[j](1);
                         c(2) = data[i](2)-centroids[j](2);    
                         DD[j] = norm(c);
                         }
                         minindex = min_element(DD.begin(),DD.end())-DD.begin();
                         clusters[minindex].push_back(data[i]);                           
                     }
                 delta = 0.0;
                 for (int i = 0; i < centroids.size(); i++){
                     b = mean(clusters[i]);
                     new_centroids[i](0) = b(0); 
                     new_centroids[i](1) = b(1); 
                     new_centroids[i](2) = b(2); 
                     b(0) = new_centroids[i](0)-centroids[i](0);
                     b(1) = new_centroids[i](1)-centroids[i](1);
                     b(2) = new_centroids[i](2)-centroids[i](2);                     
                     delta = delta + norm(b);
                     }
                 centroids = new_centroids;
             }
     }

return centroids;
}

vector<Vec3b> images_to_vector(Mat& images)
{
    vector<Vec3b> pixels;
        for(size_t i = 0; i <images.rows; i++)
            for(size_t j = 0; j < images.cols; j++)
                pixels.push_back(images.at<Vec3b>(i,j));
    return pixels;
}

void clr(){
 printf("\033[2J\033[1;1f");
}

string s="";

void function1(const sensor_msgs::ImageConstPtr& msg)
{
ROS_INFO("Imagenes recibidas: ");
    try
    {
        
        vector<String> yaml_files;
        cv::Mat imagen = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::imshow("Imagen a analizar: ", imagen);
        cv::waitKey(30);
        vector<Vec3b> data = images_to_vector(imagen);
        vector<Vec3f> centroids = get_centroids(data, 8, 1, 1);
        vector<Vec3f> lcentroids;
        Vec3f lc(0,0,0), tc(0,0,0);
        int minindex =0;
        float delta = 0;
        cout << "Reading file centroids: " << endl;
        glob("patterns/*.yaml", yaml_files, false);
        cout << "Number of yaml files: " << yaml_files.size() << endl;
        vector<float > D(yaml_files.size(),0);
        vector<float > deltas(centroids.size(),0);
        float aa, bb;
        for(size_t i = 0; i < yaml_files.size(); i++){
            FileStorage fr(yaml_files[i], FileStorage::READ);           
            fr["centroids"] >> lcentroids;
            
            for(int i = 0; i<lcentroids.size(); i++)
                delta = delta + norm(centroids[i] - lcentroids[i]);
                
            D[i] = delta;           
            delta = 0;
            fr.release();
        
        cout << "D["<< i << "] = " << D[i] << "  " << yaml_files[i] << endl;
        }
        minindex = min_element(D.begin(),D.end())-D.begin();
        cout << "Index min: " << minindex << endl; 
        s = yaml_files[minindex]; 
        cout << "Identificado como: " << s << endl;      
        
        waitKey(30);   
             
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

}

int main(int argc, char** argv)
{
    
    ros::init(argc, argv, "Imagen_recibida");
    ros::NodeHandle nodoReceptor, nodoEmisor;    
    image_transport::ImageTransport it(nodoReceptor);
    image_transport::Subscriber sub = it.subscribe("ImagenAConsultar", 1, function1);
    ros::Publisher publicadorNombre = nodoEmisor.advertise<tareas::mensaje>("Nombre", 0);
    tareas::mensaje nombre;
    nombre.nimagen = s;
    cout << "nombre: " << s << endl;
    ros::Duration seconds_sleep(1);
    while(ros::ok()){
    ROS_INFO("nodo_receptor creado y registrado");
    nombre.nimagen = s;
    publicadorNombre.publish(nombre);
    ros::spinOnce();
    seconds_sleep;
    }
    return 0;
     
}
