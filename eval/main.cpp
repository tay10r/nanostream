#include "stb_image.h"
#include "stb_image_write.h"

#include <nanostream.h>

#include <cstdio>
#include <cstdlib>

namespace {

void
process_image(const stbi_uc* rgb, const int w, const int h, const char* output_filename)
{
  const int x_tiles = w / NANOSTREAM_TILE_WIDTH;
  const int y_tiles = h / NANOSTREAM_TILE_HEIGHT;
  const int num_tiles = x_tiles * y_tiles;

  auto* out_rgb = static_cast<stbi_uc*>(malloc(num_tiles * NANOSTREAM_TILE_WIDTH * NANOSTREAM_TILE_HEIGHT * 3));

  for (int i = 0; i < num_tiles; i++) {

    const int x_t = i % x_tiles;
    const int y_t = i / x_tiles;

    const int x = x_t * NANOSTREAM_TILE_WIDTH;
    const int y = y_t * NANOSTREAM_TILE_HEIGHT;

    unsigned char packet_buffer[NANOSTREAM_PACKET_SIZE];

    nanostream_encode_tile(rgb + (y * w + x) * 3, w * 3, packet_buffer);

    unsigned char* out_ptr = out_rgb + (y * (x_tiles * NANOSTREAM_TILE_WIDTH) + x) * 3;

    nanostream_decode_tile(packet_buffer, x_tiles * NANOSTREAM_TILE_WIDTH * 3, out_ptr);
  }

  stbi_write_png(output_filename,
                 x_tiles * NANOSTREAM_TILE_WIDTH,
                 y_tiles * NANOSTREAM_TILE_HEIGHT,
                 3,
                 out_rgb,
                 x_tiles * NANOSTREAM_TILE_WIDTH * 3);

  free(out_rgb);
}

} // namespace

auto
main(int argc, char** argv) -> int
{
  if (argc <= 1) {
    fprintf(stderr, "usage: %s <input-filename> [output-filename]\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* input_filename = argv[1];

  const char* output_filename = (argc > 2) ? argv[2] : "result.png";

  int w = 0;
  int h = 0;
  stbi_uc* rgb = stbi_load(input_filename, &w, &h, nullptr, 3);
  if (!rgb) {
    fprintf(stderr, "failed to load \"%s\"\n", input_filename);
    return EXIT_FAILURE;
  }

  if (((w % 160) != 0) || ((h % 120) != 0)) {
    fprintf(stderr,
            "warning: image size (%dx%d) is not divisible into tile size (%dx%d)\n",
            w,
            h,
            NANOSTREAM_TILE_WIDTH,
            NANOSTREAM_TILE_HEIGHT);
  }

  process_image(rgb, w, h, output_filename);

  stbi_image_free(rgb);

  return EXIT_SUCCESS;
}
