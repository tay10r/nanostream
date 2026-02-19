#include "nanostream.h"

#include <math.h>
#include <string.h>

#define NUM_VALUES_PER_BLOCK 192
#define NUM_EIGEN_VALUES 8
#define BLOCK_SIZE 8
#define BYTES_PER_EV_BLOCK 4
#define BLOCKS_PER_X (NANOSTREAM_TILE_WIDTH / BLOCK_SIZE)
#define BLOCKS_PER_Y (NANOSTREAM_TILE_HEIGHT / BLOCK_SIZE)

extern const float nanostream_mean[NUM_VALUES_PER_BLOCK];
extern const float nanostream_eigen_values[NUM_EIGEN_VALUES][NUM_VALUES_PER_BLOCK];

static float
u8_to_f32(const unsigned char x)
{
  return ((float)x) * (1.0F / 255.0F);
}

static unsigned char
f32_to_u8(const float x)
{
  float y = x;
  if (y < 0.0F)
    y = 0.0F;
  if (y > 1.0F)
    y = 1.0F;
  int v = (int)lrintf(y * 255.0F);
  if (v < 0)
    v = 0;
  if (v > 255)
    v = 255;
  return (unsigned char)v;
}

static void
block_to_vec(const unsigned char* rgb, const int pitch, float* v)
{
  float* r = v;
  float* g = r + (BLOCK_SIZE * BLOCK_SIZE);
  float* b = g + (BLOCK_SIZE * BLOCK_SIZE);
  for (int y = 0; y < BLOCK_SIZE; y++) {
    const unsigned char* line = rgb + y * pitch;
    for (int x = 0; x < BLOCK_SIZE; x++) {
      r[y * BLOCK_SIZE + x] = u8_to_f32(line[x * 3 + 0]);
      g[y * BLOCK_SIZE + x] = u8_to_f32(line[x * 3 + 1]);
      b[y * BLOCK_SIZE + x] = u8_to_f32(line[x * 3 + 2]);
    }
  }
}

/* FIXED: center by mean, accumulate, and store. */
static void
to_eigen_values(const float* v, float* eigen_values_out)
{
  for (int i = 0; i < NUM_EIGEN_VALUES; i++) {
    float s = 0.0F;
    for (int j = 0; j < NUM_VALUES_PER_BLOCK; j++) {
      const float centered = v[j] - nanostream_mean[j];
      s += centered * nanostream_eigen_values[i][j];
    }
    eigen_values_out[i] = s;
  }
}

static void
expand_eigen_value_bounds(const float* eigen_values, float* ev_min, float* ev_max)
{
  for (int i = 0; i < NUM_EIGEN_VALUES; i++) {
    const float v = eigen_values[i];
    ev_min[i] = fminf(ev_min[i], v);
    ev_max[i] = fmaxf(ev_max[i], v);
  }
}

static int
quantize_f32(const float x, const float min_x, const float max_x, const int res)
{
  if (res <= 0)
    return 0;

  const float denom = (max_x - min_x);
  if (!(denom > 0.0F))
    return 0;

  float t = (x - min_x) / denom;
  if (t < 0.0F)
    t = 0.0F;
  if (t > 1.0F)
    t = 1.0F;

  int q = (int)lrintf(t * (float)res);
  if (q < 0)
    q = 0;
  if (q > res)
    q = res;
  return q;
}

static float
dequantize_f32(const int q, const float min_x, const float max_x, const int res)
{
  if (res <= 0)
    return min_x;

  int qq = q;
  if (qq < 0)
    qq = 0;
  if (qq > res)
    qq = res;

  const float t = ((float)qq) * (1.0F / (float)res);
  return min_x + t * (max_x - min_x);
}

/* FIXED packing: [8,8,4,4,2,2,2,2] bits into 4 bytes. */
static void
quantize_eigen_values(const float* ev, const float* ev_min, const float* ev_max, unsigned char* bits)
{
  const int q0 = quantize_f32(ev[0], ev_min[0], ev_max[0], 255);
  const int q1 = quantize_f32(ev[1], ev_min[1], ev_max[1], 255);
  const int q2 = quantize_f32(ev[2], ev_min[2], ev_max[2], 15);
  const int q3 = quantize_f32(ev[3], ev_min[3], ev_max[3], 15);

  const int q4 = quantize_f32(ev[4], ev_min[4], ev_max[4], 3);
  const int q5 = quantize_f32(ev[5], ev_min[5], ev_max[5], 3);
  const int q6 = quantize_f32(ev[6], ev_min[6], ev_max[6], 3);
  const int q7 = quantize_f32(ev[7], ev_min[7], ev_max[7], 3);

  bits[0] = (unsigned char)(q0 & 0xFF);
  bits[1] = (unsigned char)(q1 & 0xFF);
  bits[2] = (unsigned char)(((q2 & 0x0F) << 4) | (q3 & 0x0F));
  bits[3] = (unsigned char)((q4 & 0x03) | ((q5 & 0x03) << 2) | ((q6 & 0x03) << 4) | ((q7 & 0x03) << 6));
}

void
nanostream_encode_tile(const unsigned char* rgb, const int pitch, unsigned char* packet_buffer)
{
  float v[NUM_VALUES_PER_BLOCK];
  float eigen_values[BLOCKS_PER_X * BLOCKS_PER_Y][NUM_EIGEN_VALUES];
  float ev_min[NUM_EIGEN_VALUES];
  float ev_max[NUM_EIGEN_VALUES];

  for (int i = 0; i < NUM_EIGEN_VALUES; i++) {
    ev_min[i] = INFINITY;
    ev_max[i] = -INFINITY;
  }

  for (int block_y = 0; block_y < BLOCKS_PER_Y; block_y++) {
    for (int block_x = 0; block_x < BLOCKS_PER_X; block_x++) {
      const unsigned char* block_rgb_ptr = rgb + (block_y * BLOCK_SIZE) * pitch + (block_x * BLOCK_SIZE * 3);
      block_to_vec(block_rgb_ptr, pitch, v);
      float* ev = eigen_values[block_y * BLOCKS_PER_X + block_x];
      to_eigen_values(v, ev);
      expand_eigen_value_bounds(ev, ev_min, ev_max);
    }
  }

  memcpy(packet_buffer, ev_min, sizeof(ev_min));
  packet_buffer += sizeof(ev_min);

  memcpy(packet_buffer, ev_max, sizeof(ev_max));
  packet_buffer += sizeof(ev_max);

  for (int i = 0; i < BLOCKS_PER_X * BLOCKS_PER_Y; i++) {
    quantize_eigen_values(eigen_values[i], ev_min, ev_max, packet_buffer);
    packet_buffer += BYTES_PER_EV_BLOCK;
  }
}

static void
eigen_values_to_block_vec(const float* ev, float* v_out)
{
  for (int j = 0; j < NUM_VALUES_PER_BLOCK; j++) {
    float x = nanostream_mean[j];
    for (int i = 0; i < NUM_EIGEN_VALUES; i++) {
      x += ev[i] * nanostream_eigen_values[i][j];
    }
    v_out[j] = x;
  }
}

static void
vec_to_block(unsigned char* rgb, const int pitch, const float* v)
{
  const float* r = v;
  const float* g = r + (BLOCK_SIZE * BLOCK_SIZE);
  const float* b = g + (BLOCK_SIZE * BLOCK_SIZE);

  for (int y = 0; y < BLOCK_SIZE; y++) {
    unsigned char* line = rgb + y * pitch;
    for (int x = 0; x < BLOCK_SIZE; x++) {
      const int idx = y * BLOCK_SIZE + x;
      line[x * 3 + 0] = f32_to_u8(r[idx]);
      line[x * 3 + 1] = f32_to_u8(g[idx]);
      line[x * 3 + 2] = f32_to_u8(b[idx]);
    }
  }
}

void
nanostream_decode_tile(const unsigned char* packet_buffer, int pitch, unsigned char* rgb)
{
  float ev_min[NUM_EIGEN_VALUES];
  float ev_max[NUM_EIGEN_VALUES];

  memcpy(ev_min, packet_buffer, sizeof(ev_min));
  packet_buffer += sizeof(ev_min);

  memcpy(ev_max, packet_buffer, sizeof(ev_max));
  packet_buffer += sizeof(ev_max);

  float ev[NUM_EIGEN_VALUES];
  float v[NUM_VALUES_PER_BLOCK];

  for (int block_y = 0; block_y < BLOCKS_PER_Y; block_y++) {
    for (int block_x = 0; block_x < BLOCKS_PER_X; block_x++) {
      const unsigned char b0 = packet_buffer[0];
      const unsigned char b1 = packet_buffer[1];
      const unsigned char b2 = packet_buffer[2];
      const unsigned char b3 = packet_buffer[3];
      packet_buffer += BYTES_PER_EV_BLOCK;

      const int q0 = (int)b0;
      const int q1 = (int)b1;
      const int q2 = (int)((b2 >> 4) & 0x0F);
      const int q3 = (int)(b2 & 0x0F);

      const int q4 = (int)(b3 & 0x03);
      const int q5 = (int)((b3 >> 2) & 0x03);
      const int q6 = (int)((b3 >> 4) & 0x03);
      const int q7 = (int)((b3 >> 6) & 0x03);

      ev[0] = dequantize_f32(q0, ev_min[0], ev_max[0], 255);
      ev[1] = dequantize_f32(q1, ev_min[1], ev_max[1], 255);
      ev[2] = dequantize_f32(q2, ev_min[2], ev_max[2], 15);
      ev[3] = dequantize_f32(q3, ev_min[3], ev_max[3], 15);
      ev[4] = dequantize_f32(q4, ev_min[4], ev_max[4], 3);
      ev[5] = dequantize_f32(q5, ev_min[5], ev_max[5], 3);
      ev[6] = dequantize_f32(q6, ev_min[6], ev_max[6], 3);
      ev[7] = dequantize_f32(q7, ev_min[7], ev_max[7], 3);

      eigen_values_to_block_vec(ev, v);

      unsigned char* block_rgb_ptr = rgb + (block_y * BLOCK_SIZE) * pitch + (block_x * BLOCK_SIZE * 3);
      vec_to_block(block_rgb_ptr, pitch, v);
    }
  }
}
