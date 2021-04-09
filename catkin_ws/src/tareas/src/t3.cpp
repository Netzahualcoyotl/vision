#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#define N 1
using namespace cv;
using namespace std;

vector<Vec3d> disturb_centroids(vector<Vec3d> centroids, double epsilon)
{
vector<Vec3d> new_centroids;
Vec3d v;
    for(int i = 0; i < centroids.size(); i++){
        v(0) = ((double) rand()) / (double) RAND_MAX; 
        v(1) = ((double) rand()) / (double) RAND_MAX;
        v(2) = ((double) rand()) / (double) RAND_MAX;
        v = epsilon * v / norm(v);
        new_centroids.push_back(centroids[i] + v);
        new_centroids.push_back(centroids[i] - v);
        }

return new_centroids;

}

vector<Vec3d> get_centroids(vector<Vec3b> dat, double m, double epsilon, double tol)
{
vector<Vec3d> centroids, e((0,0,0)),data(dat.size(),(0,0,0));
Scalar b = mean(dat);
Vec3d p(b(0),b(1),b(2)),c;
centroids.push_back(p);

cout << centroids[0] << endl;

for(int i = 0; i < dat.size(); i++) {
    data[i](0) = dat[i](0);
    data[i](1) = dat[i](1);
    data[i](2) = dat[i](2);
    }

double delta = tol + 1;
int minindex=0;


 while (centroids.size() < m)
     {
         centroids = disturb_centroids(centroids, epsilon);
         
         cout <<"Numero de centroides: "<< centroids.size() << endl;
         delta = tol + 1;
         while (delta > tol)
             {
                 vector<vector<Vec3d> > clusters(centroids.size(), e);
                 vector<double> DD(centroids.size(),0);
                 vector<Vec3d> new_centroids(centroids.size(),(0,0,0));
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
                 cout << "Current delta : " << delta << endl;
             }
     }

return centroids;
}

vector<Vec3b> images_to_vector(vector<Mat>& images)
{
    vector<Vec3b> pixels;
    for(size_t k = 0; k < images.size(); k++)
        for(size_t i = 0; i <images[k].rows; i++)
            for(size_t j = 0; j < images[k].cols; j++)
                pixels.push_back(images[k].at<Vec3b>(i,j));
    return pixels;
}

int main(int argc, char** argv)
{
    vector<Mat> images;
    vector<String> img_files;
    string pattern(argv[1]);
    vector<Vec3d> lcentroids;
    glob("dataset/training/" + pattern + "/*.jpg", img_files, false);
    for(size_t i = 0; i < img_files.size(); i++)
        images.push_back(imread(img_files[i]));
    cout << "Getting centroids for " << pattern << endl;
    vector<Vec3b> data = images_to_vector(images);
    vector<Vec3d> centroids = get_centroids(data, 8, 1, 1);
    for(int i = 0; i<centroids.size(); i++) cout << centroids[i] << endl;
    cout << "Storing centroids for " << pattern << endl;
    FileStorage fs("patterns/" + pattern + ".yaml", FileStorage::WRITE);
    fs << "centroids" << centroids;
    fs.release();
    cout << "Reading file centroids: " << endl;
    FileStorage fr("patterns/" + pattern + ".yaml", FileStorage::READ);
    fr["centroids"] >> lcentroids;
    for(int i = 0; i<lcentroids.size(); i++) cout << lcentroids[i] << endl;
    fr.release();

 return 0;
 
}
