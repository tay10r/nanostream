#pragma once

#define NANOSTREAM_TILE_WIDTH 160

#define NANOSTREAM_TILE_HEIGHT 120

#define NANOSTREAM_PACKET_SIZE (1200 + (12 * sizeof(float) * 2))

#ifdef __cplusplus
extern "C"
{
#endif

  void nanostream_encode_tile(const unsigned char* rgb, int pitch, unsigned char* packet_buffer);

  void nanostream_decode_tile(const unsigned char* packet_buffer, int pitch, unsigned char* rgb);

#ifdef __cplusplus
} /* extern "C" */
#endif
