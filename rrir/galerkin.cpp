#include <iostream>
#include <fstream>
#include <math.h>
#include <set>
#include <functional>
#include <Eigen/Dense>
#include <matplotlibcpp.h>
using namespace std;
namespace plt = matplotlibcpp;

double X[2] = {1.0/sqrt(3), -1.0/sqrt(3)};
int n;
double h;

double e(int i, double x) {
    double left_point = (i-1)*h;
    double right_point = (i+1)*h;
    if (x < left_point || x > right_point || x < 0.0 || x > 2.0)
        return 0.0;
    if (x < (right_point+left_point)/2.0)
        return (x-left_point)/h;
    else
        return (right_point-x)/h;
}

double e_derivative(int i, double x) {
    double left_point = (i-1)*h;
    double right_point = (i+1)*h;
    if (x < left_point || x > right_point || x < 0.0 || x > 2.0)
        return 0.0;
    if (x < (right_point+left_point)/2.0)
        return 1.0/h;
    else
        return -1.0/h;
}

double gauss_quadrature(function<double(double)> f, double a, double b) { 
    double result = 0.0;
    for (int i = 0; i < 2; i++) 
        result += f(X[i]*(b-a)/2+(b+a)/2);
    return result*(b-a)/2;
}

double B(int i, int j) {
    if (abs(i-j)>1) //trojkaty z funkcji testowej nie maja czesci wspolnej, wszystkie mnozenia ei*ej wyjda 0
        return 0.0;

    double ans = 0;
    set<pair<double, double>> domain;

    if(i!=0)
        domain.insert({(i-1)*h, i*h});
    if(i!=n)
        domain.insert({i*h, (i+1)*h});
    if(j!=0)
        domain.insert({(j-1)*h, j*h});
    if(j!=n)
        domain.insert({j*h, (j+1)*h});

    for(auto [left, right] : domain) {
        ans += (gauss_quadrature([i, j](double x){return e_derivative(i, x) * e_derivative(j, x);}, left, right)
    -gauss_quadrature([i, j](double x){return e(i, x) * e(j, x);}, left, right));
    }

    return ans-e(i, 2)*e(j, 2); 
}

double L(int i) { 
    set<pair<double, double>> domain;
    if(i!=0) 
        domain.insert({(i-1)*h, i*h});
    if(i!=n)
        domain.insert({i*h, (i+1)*h});

    double ans = 0;
    for(auto [left, right] : domain)
        ans += gauss_quadrature([i](double x){return sin(x) * e(i, x);}, left, right);

    return ans;
}

int main() {
    cin >> n; //l. przedzialow

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_B(n, n); // n, bo element 0 nie istnieje - warunek Dirichleta
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_L(n, 1);

    for (int i=0; i<n; i++) {
        matrix_L(i, 0) = 0.0; 
        for (int j=0; j<n; j++)
            matrix_B(i, j) = 0.0;
    }

    h = 2.0/((double)n); //dlugosc przedzialu dzielimy na n elementow
    for (int i=1; i<=n; i++) {
        matrix_L(i-1, 0) = L(i); 
        for (int j=i-1; j<=i+1; j++)
            if(j>0 && j<=n)
                matrix_B(i-1, j-1) = B(i, j);
    }

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++)
            cout << matrix_B(i, j) << ' ';
        cout << endl;
    }
    cout << endl;
    for (int i=0; i<n; i++)
        cout << matrix_L(i) << ' ';
    cout << endl;
    
    Eigen::Matrix<double, Eigen::Dynamic, 1> u = matrix_B.colPivHouseholderQr().solve(matrix_L);

    for (int i=0; i<n; i++)
        cout << u[i] << ' ';

    ofstream fileA("results.txt", ios::out);
    fileA << n << endl;
    for (int i = 0; i < n; i++)
        fileA << h*i << ";" << u[i] <<endl;

    fileA.close();

    vector<double> x_values, y_values;
    for(double i=0.0; i<=2.0; i+=0.01) {
        x_values.push_back(i);
        double ans = 0;
        for(int j=1;j<=n;j++)
            ans += e(j, i) * u[j-1];
        y_values.push_back(ans);
    }

    plt::plot(x_values, y_values);
    plt::show();
    plt::detail::_interpreter::kill();
    return 0;
}