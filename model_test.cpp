#include <assert.h>
#include "hdf5.h"

#define DATA "test100.hdf5"
#define MODEL "model.hdf5"

#define BATCH_SIZE 99
#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

int xdims[4] = {BATCH_SIZE, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
int rdims[2] = {BATCH_SIZE, NUM_DIGITS};

int conv1dims[4] = {5, 5, 1, 32};
int conv2dims[4] = {5, 5, 32, 64};
int fc1dims[2] = {1024, 128};
int fc2dims[2] = {128, 10};

void loadData(float *x, float *y) {
   hid_t file_id, x_id, y_id; // identifiers
   herr_t status;

   // Open the data file
   file_id = H5Fopen(DATA, H5F_ACC_RDWR, H5P_DEFAULT);

   // Open the dataset x and y
   x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
   y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

   // Get the dataset x dimensions
   hid_t xspace = H5Dget_space(x_id); 
   const int xndims = H5Sget_simple_extent_ndims(xspace); // 4
   assert(xndims == 4);
   hsize_t xdims[xndims];
   H5Sget_simple_extent_dims(xspace, xdims, NULL);
   printf("xdims %d x %d x %d x %d\n", (int)(xdims[0]), (int)(xdims[1]), (int)(xdims[2]), (int)(xdims[3]));

   // Read the dataset x and y
   status = H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
   assert(status >= 0);
   status = H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y);
   assert(status >= 0);

   // Close the dataset x and y
   status = H5Dclose(x_id);
   status = H5Dclose(y_id);

   // Close the file
   status = H5Fclose(file_id);
}

void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
   hid_t file_id, conv1_id, conv2_id, fc1_id, fc2_id;
   herr_t status;

   // Open the model file
   file_id = H5Fopen(MODEL, H5F_ACC_RDWR, H5P_DEFAULT);

   // Open the dataset
   conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
   conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
   fc1_id = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
   fc2_id = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

   // Read the dataset
   status = H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv1);
   assert(status >= 0);
   status = H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv2);
   assert(status >= 0);
   status = H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1);
   assert(status >= 0);
   status = H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2);
   assert(status >= 0);

   // Close the dataset x and y
   status = H5Dclose(conv1_id);
   status = H5Dclose(conv2_id);
   status = H5Dclose(fc1_id);
   status = H5Dclose(fc2_id);

   // Close the file
   status = H5Fclose(file_id);
}

// Convolution layer
void conv_forward_valid(float *X, int xdims[4], float* W, int wdims[4], float *Y, int ydims[4]) {
   int batch_size = xdims[0], num_rows = xdims[1], num_cols = xdims[2];
   int filter_h = wdims[0], filter_w = wdims[1], in_channel = wdims[2], out_channel = wdims[3];
   int i, m, w, h, p, q, c;

   for (i = 0; i < ydims[0]; ++i)
      for (m = 0; m < ydims[3]; ++m)
         for (w = 0; w < ydims[2]; ++w)
            for (h = 0; h < ydims[1]; ++h)
               for (p = 0; p < filter_h; ++p)
                  for (q = 0; q < filter_w; ++q)
                     for (c = 0; c < in_channel; ++c)
                        Y[i * ydims[1] * ydims[2] * ydims[3] + h * ydims[2] *ydims[3] + w * ydims[3] + m] 
                           += X[i * xdims[1] * xdims[2] * xdims[3] + (h + p) * xdims[2] * xdims[3] + (w + q) * xdims[3] + c] 
                              * W[p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m];
}

// Recified linear unit 4d
void relu4(float *X, int xdims[4]) {
   for (int i = 0; i < xdims[0]*xdims[1]*xdims[2]*xdims[3]; ++i)
      X[i] = (X[i] < 0) ? 0 : X[i];
}


// Recified linear unit 2d
void relu2(float *X, int xdims[2]) {
   for (int i = 0; i < xdims[0]*xdims[1]; ++i)
      X[i] = (X[i] < 0) ? 0 : X[i];
}

void average_pool(float *X, int xdims[4], int pool_size, float *Y, int ydims[4]) {
   int batch_size = xdims[0], H = xdims[1], W = xdims[2], M = xdims[3];
   int i, m, w, h, p, q;

   for (i = 0; i < ydims[0]; ++i)
      for (m = 0; m < ydims[3]; ++m)
         for (w = 0; w < ydims[2]; ++w)
            for (h = 0; h < ydims[1]; ++h)
               for (p = 0; p < pool_size; ++p)
                  for (q = 0; q < pool_size; ++q)
                     Y[i * ydims[1] * ydims[2] * ydims[3] + h * ydims[2] *ydims[3] + w * ydims[3] + m] 
                        += X[i * xdims[1] * xdims[2] * xdims[3] + (pool_size * h + p) * xdims[2] * xdims[3] + (pool_size * w + q) * xdims[3] + m] 
                           / (1.0 * pool_size*pool_size);
}

void fully_forward(float *X, int xdims[2], float *W, int wdims[2], float *Y, int ydims[2]) {
   int i, j, k;
   float sum;
   
   for (i = 0; i < xdims[0]; ++i) {
      for (j = 0; j < wdims[1]; ++j) {
         sum = 0;
         for(k = 0; k < xdims[1]; ++k) {
            sum += X[i * xdims[1] + k] * W[k * wdims[1] + j]; 
         }
         Y[i * wdims[1] + j] = sum;
      }
   }
}

void argmax(float *X, int xdims[2], int *Y) {
   for (int i = 0; i < xdims[0]; ++i) {
      int max_idx = 0;
      float max = X[i * xdims[1]];
      for (int j = 0; j < xdims[1]; ++j) {
         if (X[i * xdims[1] + j] > max)
            max_idx = j;
      }
      Y[i] = max_idx;
   }
}

void forward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {

   int adims[4] = { xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3] };
   float *a = (float *)calloc(adims[0]* adims[1] * adims[2] * adims[3], sizeof(float));
   conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
   
   relu4(a, adims);

   int pool_size = 2;
   int bdims[4] = { adims[0], adims[1]/pool_size, adims[2]/pool_size, adims[3] };
   float *b = (float *)calloc(bdims[0]* bdims[1] * bdims[2] * bdims[3], sizeof(float));
   average_pool(a, adims, pool_size, b, bdims);

   int cdims[4] = { bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3] };
   float *c = (float *)calloc(cdims[0]* cdims[1] * cdims[2] * cdims[3], sizeof(float));
   conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);
   
   relu4(c, cdims);

   int ddims[4] = { cdims[0], cdims[1]/pool_size, cdims[2]/pool_size, cdims[3] };
   float *d = (float *)calloc(ddims[0]* ddims[1] * ddims[2] * ddims[3], sizeof(float));
   average_pool(c, cdims, pool_size, d, ddims);

   int edims[2] = { ddims[0], fc1dims[1] };
   float *e = (float *)calloc(edims[0]* edims[1], sizeof(float));
   fully_forward(d, ddims, fc1, fc1dims, e, edims);

   relu2(e, edims);

   int fdims[2] = { edims[0], fc2dims[1] };
   float *f = (float *)calloc(fdims[0]* fdims[1], sizeof(float));
   fully_forward(e, edims, fc2, fc2dims, f, fdims);

   argmax(f, fdims, out);

   free(a);
   free(b);
   free(c);
   free(d);
   free(e);
   free(f);
}

int main() {

   // Load data into x and y
   float *x = (float *)malloc(BATCH_SIZE*NUM_ROWS*NUM_COLS*NUM_CHANNELS*sizeof(float));
   float *r = (float *)malloc(BATCH_SIZE*NUM_DIGITS*sizeof(float));
   loadData(x, r);

   // Load model
   float *conv1 = (float *)malloc(5*5*1*32*sizeof(float));
   float *conv2 = (float *)malloc(5*5*32*64*sizeof(float));
   float *fc1 = (float *)malloc(1024*128*sizeof(float));
   float *fc2 = (float *)malloc(128*10*sizeof(float));
   loadModel(conv1, conv2, fc1, fc2);

   int *out = (int *)calloc(BATCH_SIZE, sizeof(int));
   forward_operation(x, conv1, conv2, fc1, fc2, out);

   int *ref = (int *)calloc(BATCH_SIZE, sizeof(int));
   argmax(r, rdims, ref);

   int num_correct = 0;
   for (int i = 0; i < BATCH_SIZE; ++i) {
      if (out[i] == r[i])
         num_correct++;
   }
   printf("Done. Correctness: %f\n", (float)num_correct/BATCH_SIZE);

   free(x);
   free(r);
   free(conv1);
   free(conv2);
   free(fc1);
   free(fc2);
   free(out);

   return 0;
}