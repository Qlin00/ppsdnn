#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#include "mkl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>

using namespace std;


int output_dim[5000];
int COUNT = 0;
int ALIGN = 256;

void get_sparse_matrix(string filename, sparse_matrix_t &A, MKL_INT &N, MKL_INT &M, MKL_INT &NNZ, \
MKL_INT * &rows_start, MKL_INT *& rows_end, MKL_INT *& col_indx, double *& values){
    ifstream infile;
	infile.open(filename.c_str(), ios::in);
	if (!infile.is_open()){
		cout << "读取文件失败" << endl;
		exit(1);
	}
	string x;
    getline(infile, x);
    istringstream ss(x);
    string temp;
    getline(ss, temp, ','); N = atoi(temp.c_str());
    getline(ss, temp, ','); M = atoi(temp.c_str());
    getline(ss, temp, ','); NNZ = atoi(temp.c_str());
    // NNZ = 2048;
    rows_start = (MKL_INT *) mkl_calloc(N, sizeof(MKL_INT), ALIGN);
    rows_end = (MKL_INT *) mkl_calloc(N, sizeof(MKL_INT), ALIGN);
    col_indx = (MKL_INT *) mkl_calloc(NNZ, sizeof(MKL_INT), ALIGN);
    values = (double *) mkl_calloc(NNZ, sizeof(double), ALIGN);
    getline(infile, x);
    istringstream ss1(x);
    for(int i = 0; i < N; i++){
        ss1 >> rows_start[i];
        if(i > 0)
            rows_end[i-1] = rows_start[i];
    }
    rows_end[N-1] = NNZ;

    getline(infile, x);
    istringstream ss2(x);
    for(int i = 0; i < NNZ; i++){
        ss2 >> col_indx[i];
        values[i] = rand()/(RAND_MAX+1.0);
    }

    // sparse_matrix_t       A = NULL;
    sparse_index_base_t    indexing = SPARSE_INDEX_BASE_ZERO;
    struct matrix_descr    descr_type_gen;
    descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_create_csr( &A, indexing, N, M, rows_start, rows_end, col_indx, values);

    // cout << N << "  "  << M << endl;
    // return A;
}


void get_batchsizes(){
    ifstream infile;
	infile.open("dlmc/rn50_batchsizes.txt", ios::in);
	if (!infile.is_open()){
		cout << "读取文件失败" << endl;
		return;
	}
    string line;
    while (getline(infile, line)){
		stringstream ss(line);
        string temp;
		getline(ss, temp, ',');
        ss >> output_dim[COUNT];
        COUNT += 1;
	}
}


 
void get_file_names(std::string path, std::vector<std::string>& files)
{
	DIR *dir;
	struct dirent *ptr;
	if ((dir = opendir(path.c_str())) == NULL){
		perror("Open dir error...");
		return;
	}

	while ((ptr = readdir(dir)) != NULL){
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
			continue;
		else if (ptr->d_type == 8){
			std::string strFile;
			strFile = path;
			strFile += "/";
			strFile += ptr->d_name;
			files.push_back(strFile);
		}
		else
			continue;
	}
	closedir(dir);
	return;
}



int main() {
    get_batchsizes();

    vector<string> filenames;
    string path("dlmc/rn50/random_pruning/0.5");
    get_file_names(path,filenames);
    for(int fi = 0; fi < 10; fi++){
        MKL_INT N, K, M, NNZ;
        M = output_dim[fi];
        string filename = filenames[fi];
        // string filename = "dlmc/rn50/magnitude_pruning/0.5/bottleneck_1_block_group_projection_block_group1.smtx";
        sparse_matrix_t A;
        MKL_INT *rows_start , *rows_end, *col_indx;
        double *values;
        get_sparse_matrix(filename, A, N, K, NNZ, rows_start, rows_end, col_indx, values);
        cout << filename << endl;
        cout << N << "  " << K << "  " << M << "  " << NNZ << endl;
        double *B = (double *) mkl_calloc ( K*M, sizeof(double),ALIGN);
        double *C = (double *) mkl_calloc ( N*M, sizeof(double), ALIGN);
        for(int i=0; i < K; i++) {
            for(int j = 0; j < M; j++) 
                B[i*M+j] = rand()/(RAND_MAX+1.0);
        }
        sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
        struct matrix_descr    descr_type_gen;
        descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;
        clock_t start,end;
        cout << 3 << endl;
        start = clock(); 
        for(int i = 0; i < 1000; i++)
            mkl_sparse_d_mm(operation,1.0,A,descr_type_gen,SPARSE_LAYOUT_COLUMN_MAJOR, B, M, K, 0.0, C, N);
        end = clock();
        cout << 4 << endl;
        double endtime=(double)(end-start)/CLOCKS_PER_SEC * 1000; // ms
        printf("%lf\n", endtime);

        mkl_free(rows_start);
        mkl_free(rows_end);
        mkl_free(col_indx);
        mkl_free(values);


        mkl_free(B);
        mkl_free(C);
    }

    return 0;
}