#include "nanostream.h"

#include <math.h>
#include <string.h>

#define NUM_VALUES_PER_BLOCK 192
#define NUM_EIGEN_VALUES 8
#define BLOCK_SIZE 8
#define CHUNK_SIZE 5
#define BYTES_PER_EV_BLOCK 4
#define BLOCKS_PER_X (NANOSTREAM_TILE_WIDTH / BLOCK_SIZE)
#define BLOCKS_PER_Y (NANOSTREAM_TILE_HEIGHT / BLOCK_SIZE)

extern const float nanostream_mean[NUM_VALUES_PER_BLOCK];

extern const float nanostream_eigen_values[NUM_EIGEN_VALUES][NUM_VALUES_PER_BLOCK];

static float u8_to_f32(const unsigned char x)
{
    return ((float)x) * (1.0F / 255.0F);
}

static void block_to_vec(const unsigned char* rgb, const int pitch, float* v)
{
    float* r = v;
    float* g = r + (BLOCK_SIZE * BLOCK_SIZE);
    float* b = g + (BLOCK_SIZE * BLOCK_SIZE);
    for (int y = 0; y < BLOCK_SIZE; y++) {
        const unsigned char* line = rgb + y * pitch;
        for (int x = 0; x < BLOCK_SIZE; x++) {
            r[y * BLOCK_SIZE + x]  = u8_to_f32(line[x * 3 + 0]);
            g[y * BLOCK_SIZE + x]  = u8_to_f32(line[x * 3 + 1]);
            b[y * BLOCK_SIZE + x]  = u8_to_f32(line[x * 3 + 2]);
        }
    }
}

static void to_eigen_values(const float* v, float* eigen_values)
{
    for (int i = 0; i < NUM_EIGEN_VALUES; i += 4) {
        /* if we do a rain dance the compiler might vectorize this properly. */
        float s[4] = { 0, 0, 0, 0 };
        for (int j = 0; j < NUM_VALUES_PER_BLOCK; j++) {
            const float v_j = v[j] - nanostream_mean[j];
            s[0] += v_j * nanostream_eigen_values[i + 0][j];
            s[1] += v_j * nanostream_eigen_values[i + 1][j];
            s[2] += v_j * nanostream_eigen_values[i + 2][j];
            s[3] += v_j * nanostream_eigen_values[i + 3][j];
        }
        eigen_values[i + 0] = s[0];
        eigen_values[i + 1] = s[1];
        eigen_values[i + 2] = s[2];
        eigen_values[i + 3] = s[3];
    }
}

static void expand_eigen_value_bounds(const float* eigen_values, float* ev_min, float* ev_max)
{
    for (int i = 0; i < NUM_EIGEN_VALUES; i++) {
        const float v = eigen_values[i];
        ev_min[i] = fminf(ev_min[i], v);
        ev_max[i] = fmaxf(ev_max[i], v);
    }
}

static int quantize(const float x, const float min_x, const float max_x, const int res)
{
    const int y = (int)(((min_x + x) / (min_x + max_x)) * ((float)res));
    return y;
}

static void quantize_eigen_values(const float* eigen_values, const float* ev_min, const float* ev_max, unsigned char* bits)
{
    /* each eigen value gets the following number of bits:
     *   0 1 2 3 4 5 6 7 | eigen value
     *   8 8 4 4 2 2 2 2 | bit count
     */

    bits[0] = quantize(eigen_values[0], ev_min[0], ev_max[0], 255);
    bits[1] = quantize(eigen_values[0], ev_min[1], ev_max[1], 255);

    bits[2] = (quantize(eigen_values[2], ev_min[2], ev_max[2], 15) << 4) | quantize(eigen_values[3], ev_min[3], ev_max[3], 255);

    int tmp = 0;
    tmp |= quantize(eigen_values[4], ev_min[4], ev_max[4], 3);
    tmp |= quantize(eigen_values[5], ev_min[5], ev_max[5], 3) << 2;
    tmp |= quantize(eigen_values[6], ev_min[6], ev_max[6], 3) << 4;
    tmp |= quantize(eigen_values[7], ev_min[7], ev_max[7], 3) << 6;

    bits[3] = tmp;
}

void nanostream_encode_tile(const unsigned char* rgb, const int pitch, unsigned char* packet_buffer)
{
    float v[NUM_VALUES_PER_BLOCK];
    float eigen_values[BLOCKS_PER_X * BLOCKS_PER_Y][NUM_EIGEN_VALUES];
    float ev_min[NUM_EIGEN_VALUES];
    float ev_max[NUM_EIGEN_VALUES];

    for (int i = 0; i < NUM_EIGEN_VALUES; i++) {
        ev_min[i] = -1.0e6F;
        ev_max[i] = 1.0e6F;
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

void nanostream_decode_tile(const unsigned char* packet_buffer, int pitch, unsigned char* rgb)
{
}