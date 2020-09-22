// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
//https://stackoverflow.com/questions/1727881/how-to-use-the-pi-constant-in-c
//To get PI
//https://stackoverflow.com/questions/8690567/setting-an-int-to-infinity-in-c
// Making an infinity variable
#include <limits>
#include <math.h>
#define _USE_MATH_DEFINES
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"
#include "utils.h"

// Shortcut to avoid Eigen:: and std:: everywhere, DO NOT USE IN .h
using namespace std;
using namespace Eigen;
//http://paulbourke.net/geometry/polygonmesh/
//http://what-when-how.com/advanced-methods-in-computer-graphics/mesh-processing-advanced-methods-in-computer-graphics-part-1/
//https://www.youtube.com/watch?v=P2xMqTgsgsE
class Mesh
{
public:
	Mesh(string f)
	{
		string fileName;
		int temp;
		Vector3f zero(0, 0, 0);
		ifstream F(f);
		F >> fileName>>numVertices >> numTriangles >> temp;

		for (int i = 0; i < numVertices; i++)
		{
			float x, y, z;
			F >> x >> y >> z;
			Vector3f temp(x, y, z);
			V.push_back(temp);
			NA.push_back(zero);
			NAngle.push_back(zero);
		}


		for (int i = 0; i < numTriangles; i++)
		{
			int temp2, temp3, temp4;
			F >> temp >> temp2 >> temp3 >> temp4;
			Vector3i temp(temp2, temp3, temp4);
			T.push_back(temp);
		}

		for (int i = 0; i < numTriangles; i++)
		{
			auto x = V[T[i].x()];
			auto y = V[T[i].y()];
			auto z = V[T[i].z()];
			Vector3f temp = x;
			Vector3f temp2 = y;
			Vector3f temp3 = z;

			Vector3f temp4 = temp3 - temp;
			Vector3f temp5 = temp2 - temp;

			Vector3f N = temp5.cross(temp4);
			normal.push_back(N);
		}

		for (int i = 0; i < numTriangles; i++)
		{
			NA[T[i].x()] += normal[i];
			NA[T[i].y()] += normal[i];
			NA[T[i].z()] += normal[i];
		}

		for (int i = 0; i < numTriangles; i++)
		{
			normal[i] = (normal[i] / normal[i].norm());
		}
		for (int i = 0; i < numVertices; i++)
		{
			NA[i] = NA[i] / NA[i].norm();
		}

		for (int i = 0; i < numTriangles; i++)
		{
			Vector3f temp = V[T[i].x()];
			Vector3f temp2 = V[T[i].y()];
			Vector3f temp3 = V[T[i].z()];

			Vector3f  vec1 = temp2 - temp;
			Vector3f vec2 = temp3 - temp;
			auto angle = acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
			float temp4 = angle;
			vec1 = temp - temp2;
			vec2 = temp3 - temp2;

			float temp5 = acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
			auto PI = M_PI;
			float temp6 =  PI - temp4 - temp5;

			NAngle[T[i].x()] += normal[i] * temp4;
			NAngle[T[i].y()] += normal[i] * temp5;
			NAngle[T[i].z()] += normal[i] * temp6;
		}

		for (int i = 0; i < numVertices; i++)
		{
			NAngle[i] = NAngle[i] / NAngle[i].norm();
		}

		Vector3f min(V[0].x(), V[0].y(), V[0].z());
		Vector3f max(V[0].x(), V[0].y(), V[0].z());
	/*	bounding_box.push_back(min);
		bounding_box.push_back(max);*/

		//for (int i = 0; i < numVertices; i++)
		//{
		//	//https://en.wikipedia.org/wiki/Bounding_volume
		////https://www.youtube.com/watch?v=xUszK2xNL3I TO understand what the box is 
		//	if (V[i].x() < bounding_box[0].x())
		//		bounding_box[0].x() = (V[i].x());
		//	else if (V[i].z() < bounding_box[0].z())
		//		bounding_box[0].z() = (V[i].z());

		//	else if (V[i].x() > bounding_box[1].x())
		//		bounding_box[1].x() = (V[i].x());
		//	else if (V[i].y() < bounding_box[0].y())
		//		bounding_box[0].y() = (V[i].y());
		//	else if (V[i].z() > bounding_box[1].z())
		//		bounding_box[1].z() = (V[i].z());

		//	else if (V[i].y() > bounding_box[1].y())
		//		bounding_box[1].y() = (V[i].y());
		//	

		//}

	}
	vector<Vector3f> getV()
	{
		return V;
	}
	vector<Vector3f> getN()
	{
		return normal;
	}
	vector<Vector3f> getNArea()
	{
		return NA;
	}
	vector<Vector3f> getNAngle()
	{
		return NAngle;
	}

	vector<Vector3i> getT()
	{
		return T;
	}
	int getNumT()
	{
		return numTriangles;
	}
	int getNumV()
	{
		return numVertices;
	}
	vector<Vector3f> getBoundingBox()
	{
		return bounding_box;
	}

private:
	int numVertices;
	int numTriangles;
	vector<Vector3f> V;
	vector<Vector3f > normal;
	vector<Vector3f > NA; //Normal Area
	vector<Vector3f>  NAngle;
	vector<Vector3i>  T;
	vector<Vector3f>  bounding_box;

};

//class Mesh
//{
//public :
//	Mesh(const char *filename) : V(),F()
//	{
//		ifstream in;
//		in.open(filename, ifstream::in);
//		if (in.fail()) {
//			cerr << "Failed to open " << filename << endl;
//			return;
//		}
//		string line;
//		while (!in.eof()) {
//			getline(in, line);
//			istringstream iss(line.c_str());
//			char trash;
//			if (!line.compare(0, 2, "v ")) {
//				iss >> trash;
//				Vector3i v;
//				for (int i = 0; i < 3; i++) 
//					iss >> v[i];
//				V.push_back(v);
//			}
//			else if (!line.compare(0, 2, "3 ")) {
//				Vector3d f;
//				int idx, cnt = 0;
//				iss >> trash;
//				while (iss >> idx) {
//					idx--; // in wavefront obj all indices start at 1, not zero
//					f[cnt++] = idx;
//				}
//				if (3 == cnt) faces.push_back(f);
//			}
//		}
//		std::cerr << "# v# " << verts.size() << " f# " << faces.size() << std::endl;
//
//		Vec3f min, max;
//		get_bbox(min, max);
//	}
//
//	int getVert() const
//	{
//		return (int)V.size();
//	}
//	int getFaces() const
//	{
//		return (int)F.size();
//	}
//
//	bool ray_triangle_intersect(const int &fi, const Vector3d &orig, const Vector3d &dir, float &tnear)
//	{
//		//Vector3f edge1 = point(V(fi, 1,0)) - point(V(fi, 0));
//		//Vector3f edge2 = point(V(fi, 2)) - point(V(fi, 0));
//		//
//		//Vector3d pvec = dir.cross(edge2); 
//		//float det = edge1 * pvec;
//		//if (det < 1e-5) return false;
//
//		//Vector3d tvec = orig - point(vert(fi, 0));
//		//float u = tvec * pvec;
//		//if (u < 0 || u > det) return false;
//
//		//Vector3d qvec = tvec.cross(edge1);
//		//float v = dir * qvec;
//		//if (v < 0 || u + v > det) return false;
//
//		//tnear = edge2 * qvec * (1. / det);
//		//return tnear > 1e-5;
//	}
//	const Vector3f &point(int b) const
//	{
//		bool eq1 = b >= 0;
//		bool eq2 = b < getVert();
//		assert(eq1 && eq2);
//		return V[b];
//	}
//	Vector3f &point(int b)
//	{
//		bool eq1 = b >= 0;
//		bool eq2 = b < getVert();
//		assert(eq1 && eq2);
//		return V[b];
//	} 
//	// coordinates of the vertex i
//	int vert(int Triangle, int index) const
//	{
//		auto eq1 = Triangle >= 0;
//		auto eq2 = Triangle < getFaces();
//		auto eq3 = index >= 0;
//		auto eq4 = index < 3;
//
//		assert(eq1&& eq2 && eq3 && eq4);
//		return F[Triangle][index];
//	}
//	// index of the vertex for the triangle fi and local index li
//	void get_bbox(Vector3f  & minimun, Vector3f &maximum) {
//		minimun = V[0];
//		maximum = V[0];
//		for (int i = 1; i < getVert(); ++i) {
//			for (int j = 0; j < 3; j++) {
//				auto eq1 = minimun[j];
//				auto eq2 = V[i][j];
//				auto eq3 = maximum[j];
//				minimun[j] = min(eq1, eq2);
//				maximum[j] = max(eq3, eq2);
//			}
//		}
//		std::cerr << "bbox: [" << minimun << " : " << maximum << "]" << endl;
//	}
//	// bounding box for all the vertices, including isolated ones
//private :
//	vector<Vector3f> V;
//	vector <Vector3i> F;
//	string F;
//};
//
//
////std::ostream& operator<<(std::ostream& out, Mesh &m)
////{
////	for (int i = 0; i < m.getVert(); i++) {
////		out << "v " << m.point(i) << endl;
////	}
////	for (int i = 0; i < m.getFaces(); i++) {
////		out << "f ";
////		for (int k = 0; k < 3; k++) {
////			out << (m.vert(i, k) + 1) << " ";
////		}
////		out << endl;
////	}
////	return out;
////}

class Illumination
{
public:
	Illumination(const Vector3d &p, const double(&i))
	{
		P = p;
		I = i;
	}
	Vector3d getP() const
	{
		return P;
	}
	double getI() const
	{
		return I;
	}
private:
	Vector3d P;
	double I;
};

class Ray
{
public:
	Ray(const Vector3d & O, const Vector3d  & dir)
	{
		Origin = O, Direction = dir;
	}
	auto getOrigin() const
	{
		return Origin;
	}
	auto getDirection() const
	{
		return Direction;
	}
private:
	Vector3d Origin;
	Vector3d Direction;
};

class Description
{
public:
	Description(const double &r, const Vector3d &al, const double &s)
	{

		Albedo = al; // same diffuse
		Refract = r;
		Spec = s;
	}
	Description()
	{
		Refract = 1;
		Albedo = Vector3d(1, 0, 0);

	}

	auto getAlbedo() const
	{
		return Albedo;
	}
	auto getRefract() const
	{
		return Refract;
	}

	auto getSpec() const
	{
		return Spec;
	}
	auto setAlbedo(Vector3d b)
	{
		Albedo = b;
	}
private:
	double Refract;
	Vector3d Albedo;
	double Spec;


};

class Sphere
{
public:

	Sphere(const Vector3d &c, const float &r, const Description &D)
	{
		Object = D;
		center = c;
		radius = r;
	}

	bool intersect(Ray R)
	{
		auto O = R.getOrigin() - center;
		auto D = R.getDirection();
		double DO = D.dot(O);
		double OO = O.dot(O);
		double DD = D.dot(D);

		if (DD <= 0) {
			return false;
		}

		double Delta = 4 * DO * DO - 4 * DD * (OO - radius * radius);
		double t1 = (-2 * DO - sqrt(Delta)) / (2 * DD);
		double t2 = (-2 * DO + sqrt(Delta)) / (2 * DD);

		if (t1 < t2) {
			t = t1;
		}
		else {
			t = t2;
		}
		return true;
	}
	auto getCenter()
	{
		return center;
	}
	auto getRadius()
	{
		return radius;
	}
	auto getT()
	{
		return t;
	}
	auto getDescription()
	{
		return Object;
	}

	double t = 0;
private:
	Vector3d center;
	double radius;
	Description Object;
};
//I−2(N⋅I)N
Vector3d Reflection(const Vector3d &I, const Vector3d &N)
{
	return I - 2 * N.dot(I) * N;
}
//scene intersect
bool SphereIntersect(Ray & ray, Vector3d &hit, Vector3d &N, vector<Sphere> &S, Description &Des)
{
	//scene
	Ray R = ray;


	auto minT = numeric_limits<double>::infinity();
	int minI = -1;
	for (int i = 0; i < S.size(); i++)
	{
		if (S[i].intersect(R) && S[i].getT() < minT)
		{
			minT = S[i].getT();
			minI = i;
		}
	}
	if (minI < 0) {
		return false;
	}
	else {
		hit = R.getOrigin() + R.getDirection() * minT;
		N = (hit - S[minI].getCenter()).normalized();
		Des = S[minI].getDescription();

	}
 //   double Plane = numeric_limits<double>::infinity();
	//auto dirY = R.getDirection().y();
	//auto Origin = R.getOrigin().y();

	////if (fabs(dirY) > 0)
	//{
	//	float Distance = -(Origin + 4) / dirY;
	//	Vector3d P = R.getOrigin() + R.getDirection() * Distance;
	//	if (Distance > 0 && fabs(P.x()) < 10 && P.z() < -10 && P.z() > -30 && Distance < minT)
	//	{
	//		Plane = Distance;
	//		hit = P;
	//		N = Vector3d(0, 1, 0);
	//		auto b = Reflection(N, hit);
	//	}
	//}
	return true;
}


//
//Vector3d Refraction(const Vector3d &I, const Vector3d &N, const double eta_t, const double eta_i = 1.0)
//{
//	return Vector3d(0, 0, 0); // TODO
//}
//cast ray
Vector3d Shade(const Description& material, const Illumination& light, Vector3d &normal, Vector3d & hit, Vector3d & origin)
{
	const auto e = 1e-3;
	auto l = (light.getP() - hit).normalized();
	auto lDistance= (light.getP() - hit).normalized();
	//Vector3d OriginofShadow;
	//Vector3d shadow_orig = light_Distance * normal < 0 ? hit - N * 1e-3 : hit + normal * 1e-3;
	//Vector3d ShadowHit, ShadowNormal;
	//Description tmp;
	//if (light_Distance * normal < 0)
	//	OriginofShadow = hit - (normal *e);
	//else
	//	OriginofShadow = hit + (normal *e);

	auto cosAngle = max(l.dot(normal), 0.0);
	auto diffuse = material.getAlbedo() * cosAngle;

	auto v = (origin - hit).normalized();
	auto rv = Reflection(-v, normal);
    cosAngle = std::max(l.dot(v), 0.0);
	auto specular = std::pow(cosAngle, material.getSpec()) * Vector3d(1, 1, 1);

	return (diffuse + specular) * light.getI();
}
//bool SphereIntersect(Ray & ray, Vector3d &hit, Vector3d &N, vector<Sphere> &S, Description &Des)

Vector3d Shade(const Description& material, const vector<Illumination>& lights, Vector3d normal, Vector3d hit, Vector3d origin) {
	Vector3d sum = Vector3d(0, 0, 0);
	Vector3d point, N;
	for (auto l : lights) {
		//Vector3d light_dir = (l.getP() - hit).normalize();
		//float light_distance = (l.getP() - point).norm();
		//Vector3d shadow_orig = light_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
		//Ray R = (origin, lDistance);
		//if(SphereIntersect(R,hit,normal,))
		sum += Shade(material, l, normal, hit, origin);
	}
	return sum;
}
//l0 is the origin of the ray and l is the ray direction. 
//plane can be defined as a point representing how far the plane is from the world origin and a normal (defining the orientation of the plane). Let's call this point p0 and the plane normal n. A vector can be computed from an
//y point on the plane by subtracting P0 from this point which we will call p.
//t which is a positive real number (which as usual, is the parametric
//distance from the origin of the ray to the point of interest along the ray).
bool intersectPlane(const Vector3d &N, const Vector3d &P0, Ray O, vector<Sphere> &S)
{
	for (int i = 0; i < S.size(); i++)
	{
		double d = N.dot(O.getDirection());
		if (d > 0.00001) {
			Vector3d temp = P0 - O.getOrigin();
			S[i].t = temp.dot(N) / d;
			return (S[i].t >= 0);
		}
	}


	return false;
}

void Tracing(const vector <Sphere> &S, vector<Illumination> & Light, bool useOrthoProjection, string Filename)
{
//	Mesh Bunny("bunny.off");
//	Mesh Cube("bumpy_cube.off");

	cout << "Running Ray tracing program" << std::endl;

	auto Sphere = S;

	double Width = 1000;
	double Height = 1000;
	double Dimension = Width * Height;
	vector <Vector3d> Display(Dimension);

	/*bool useOrthoProjection = true;*/

	for (unsigned W = 0; W < Width; W++)
	{
		for (unsigned L = 0; L < Height; L++)
		{
			// Convert (W,L) to (-1, 1)
			double X = (L + 0.5) / (Height / 2) - 1;
			double Y = (W + 0.5) / (Width / 2) - 1;
			const double focalLength = 0.9;

			const Vector3d origin = useOrthoProjection ? Vector3d(X, Y, focalLength) : Vector3d(0, 0, focalLength);
			const Vector3d pixelPosition = Vector3d(X, Y, 0);

			auto rayDirection = (pixelPosition - origin).normalized();

			Ray R(origin, rayDirection);
			Vector3d point = Vector3d(.07, .2, .1);
			Vector3d N;
			Description D;
			//bool intersectPlane(const Vector3d &N, const Vector3d &P0, Ray O, vector<Sphere> &S)

	/*		if (intersectPlane(N, point, R, Sphere))
			{
				Display[L + W * Width] = Vector3d(0, 0, 0);
			}*/
		    if (SphereIntersect(R, point, N, Sphere, D))
			{
				Display[L + W * Width] = Shade(D, Light, N, point, origin);
			}
			else
			{
				Display[L + W * Width] = Vector3d(40, 40, 40);
			}
		}
	}

	MatrixXd R = MatrixXd::Zero(Height, Width); // Store A
	MatrixXd G = MatrixXd::Zero(Height, Width); // Store G
	MatrixXd B = MatrixXd::Zero(Height, Width); // Store B
	MatrixXd A = MatrixXd::Zero(Height, Width); // Store A

	for (unsigned W = 0; W < Width; W++)
	{
		for (unsigned L = 0; L < Height; L++)
		{
			R(W, L) = Display[W + L * Width].x();
			G(W, L) = Display[W + L * Width].y();
			B(W, L) = Display[W + L * Width].z();
			A(W, L) = 1;
		}
	}

	write_matrix_to_png(R, G, B, A, Filename);
}

int main()
{

	vector<Sphere> S;
	Description Object1(1.0, Vector3d(0.6, 0.3, 0.1), 100);
	S.push_back(Sphere(Vector3d(0, 0, -6), 1, Object1));

	Description Object2(1.0, Vector3d(0.1, 0.9, 0.1), 100);
	S.push_back(Sphere(Vector3d(-3, 2, -4), 1, Object2));

	Description Object3(1.0, Vector3d(0.3, 0.2, 0.7), 1000);
	S.push_back(Sphere(Vector3d(4, -3, -5), 1, Object3));

	Description Object4(1.0, Vector3d(0.3, 0.2, 0.7), 1000);
	S.push_back(Sphere(Vector3d(0, -200000, -5), 1, Object4));

	vector<Illumination> L;
	L.push_back(Illumination(Vector3d(1, 0, 5), 0.8));
	L.push_back(Illumination(Vector3d(5, 5, 5), 0.7));
	L.push_back(Illumination(Vector3d(15, 15, -5), 0.6));

	Tracing(S, L,1,"part2.png");
	Tracing(S, L, 0, "part3.png");
	cout << "Program is over! Goodbye!";
	return 0;
}

