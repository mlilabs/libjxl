// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests of JxlDecoderSetImageRegion: the output of a region must be the same
// as the region of the output of the whole image, for all orientations, pixel
// formats and kinds of frames, and groups that do not contribute to the region
// must not be decoded.

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"
#include "lib/jxl/toc.h"

namespace jxl {
namespace {

struct EncodeParams {
  std::string name;
  size_t xsize = 640;
  size_t ysize = 480;
  // 3 for RGB, 4 for RGBA.
  size_t num_channels = 3;
  // Number of extra channels beside alpha.
  size_t num_extra = 0;
  // 0 for lossless.
  float distance = 1.0f;
  int64_t effort = 3;
  std::vector<std::pair<JxlEncoderFrameSettingId, int64_t>> options;
  std::vector<std::pair<JxlEncoderFrameSettingId, float>> float_options;
  JxlOrientation orientation = JXL_ORIENT_IDENTITY;
  // Number of frames of an animation; 0 for a still image.
  size_t num_frames = 0;
  // Whether the frames after the first are blended onto it.
  bool blend_frames = false;
  // Whether groups outside an output region are not decoded.
  bool skips = true;
};

std::ostream& operator<<(std::ostream& os, const EncodeParams& p) {
  return os << p.name;
}

// The extra channel with index `ec`, 16 bits per sample.
std::vector<uint8_t> ExtraChannelPixels(size_t xsize, size_t ysize, size_t ec) {
  std::vector<uint8_t> pixels(xsize * ysize * 2);
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t v = static_cast<uint16_t>((x * 97 + y * 31 + ec * 1000) % 65536);
      pixels[(y * xsize + x) * 2] = v >> 8;
      pixels[(y * xsize + x) * 2 + 1] = v & 255;
    }
  }
  return pixels;
}

std::vector<uint8_t> Encode(const EncodeParams& p) {
  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderUseContainer(enc.get(), JXL_FALSE));
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = p.xsize;
  info.ysize = p.ysize;
  info.bits_per_sample = 16;
  info.num_color_channels = 3;
  bool alpha = p.num_channels == 4;
  info.alpha_bits = alpha ? 16 : 0;
  info.num_extra_channels = (alpha ? 1 : 0) + p.num_extra;
  info.uses_original_profile = TO_JXL_BOOL(p.distance == 0);
  info.orientation = p.orientation;
  if (p.num_frames > 0) {
    info.have_animation = JXL_TRUE;
    info.animation.tps_numerator = 10;
    info.animation.tps_denominator = 1;
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &info));
  JxlColorEncoding color_encoding;
  JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));
  for (size_t ec = 0; ec < p.num_extra; ec++) {
    JxlExtraChannelInfo ec_info;
    JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_DEPTH, &ec_info);
    ec_info.bits_per_sample = 16;
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetExtraChannelInfo(
                                   enc.get(), (alpha ? 1 : 0) + ec, &ec_info));
  }

  JxlEncoderFrameSettings* settings =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                settings, JXL_ENC_FRAME_SETTING_EFFORT, p.effort));
  if (p.distance == 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  } else {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderSetFrameDistance(settings, p.distance));
  }
  for (const auto& option : p.options) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderFrameSettingsSetOption(
                                   settings, option.first, option.second));
  }
  for (const auto& option : p.float_options) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderFrameSettingsSetFloatOption(
                                   settings, option.first, option.second));
  }

  JxlPixelFormat format = {static_cast<uint32_t>(p.num_channels),
                           JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  JxlPixelFormat ec_format = {1, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  size_t num_frames = std::max<size_t>(p.num_frames, 1);
  for (size_t f = 0; f < num_frames; f++) {
    if (p.num_frames > 0) {
      JxlFrameHeader header;
      JxlEncoderInitFrameHeader(&header);
      header.duration = 1;
      if (p.blend_frames && f > 0) {
        header.layer_info.blend_info.blendmode = JXL_BLEND_BLEND;
      }
      EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameHeader(settings, &header));
    }
    std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(
        p.xsize, p.ysize, p.num_channels, static_cast<uint16_t>(f + 1));
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderAddImageFrame(settings, &format, pixels.data(),
                                      pixels.size()));
    for (size_t ec = 0; ec < p.num_extra; ec++) {
      std::vector<uint8_t> ec_pixels = ExtraChannelPixels(p.xsize, p.ysize, ec);
      EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetExtraChannelBuffer(
                                     settings, &ec_format, ec_pixels.data(),
                                     ec_pixels.size(), (alpha ? 1 : 0) + ec));
    }
  }
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> compressed(64);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();
  JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
  while (status == JXL_ENC_NEED_MORE_OUTPUT) {
    status = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, status);
  compressed.resize(next_out - compressed.data());
  return compressed;
}

struct DecodeParams {
  JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  // The region in output coordinates; empty for the whole image.
  Rect region;
  bool keep_orientation = false;
  bool use_callback = false;
  size_t num_threads = 0;
  // Whether to also output the extra channels beside alpha.
  bool extra_channels = false;
  // Whether to output the image when the DC is complete, before the end.
  bool flush_after_dc = false;
};

// The decoded image, or the images of the frames of an animation.
struct Decoded {
  bool success = false;
  size_t xsize = 0;
  size_t ysize = 0;
  // Per frame the main output and the extra channel outputs.
  std::vector<std::vector<uint8_t>> frames;
  std::vector<std::vector<std::vector<uint8_t>>> extra;
  // The output after the DC of the first frame, if flushed.
  std::vector<uint8_t> after_dc;
};

size_t BytesPerSample(JxlDataType type) {
  switch (type) {
    case JXL_TYPE_UINT8:
      return 1;
    case JXL_TYPE_UINT16:
    case JXL_TYPE_FLOAT16:
      return 2;
    default:
      return 4;
  }
}

// Row stride in bytes of `xsize` pixels of `format`.
size_t Stride(const JxlPixelFormat& format, size_t xsize) {
  size_t row = xsize * format.num_channels * BytesPerSample(format.data_type);
  return format.align > 1 ? DivCeil(row, format.align) * format.align : row;
}

struct CallbackOutput {
  std::vector<uint8_t>* pixels;
  size_t xsize;
  size_t ysize;
  size_t stride;
  size_t pixel_size;
  bool out_of_bounds = false;
};

void CopyPixels(void* opaque, size_t x, size_t y, size_t num_pixels,
                const void* pixels) {
  auto* out = static_cast<CallbackOutput*>(opaque);
  if (x + num_pixels > out->xsize || y >= out->ysize) {
    out->out_of_bounds = true;
    return;
  }
  memcpy(out->pixels->data() + y * out->stride + x * out->pixel_size, pixels,
         num_pixels * out->pixel_size);
}

Decoded Decode(const std::vector<uint8_t>& data, const DecodeParams& p) {
  Decoded result;
  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  JxlThreadParallelRunnerPtr runner;
  if (p.num_threads > 0) {
    runner = JxlThreadParallelRunnerMake(nullptr, p.num_threads);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetParallelRunner(dec.get(), JxlThreadParallelRunner,
                                          runner.get()));
  }
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetKeepOrientation(
                                 dec.get(), TO_JXL_BOOL(p.keep_orientation)));
  int events = JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE;
  if (p.flush_after_dc) {
    events |= JXL_DEC_FRAME_PROGRESSION;
    EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetProgressiveDetail(dec.get(), kDC));
  }
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSubscribeEvents(dec.get(), events));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetInput(dec.get(), data.data(), data.size()));
  JxlDecoderCloseInput(dec.get());

  JxlBasicInfo info;
  std::vector<uint8_t> pixels;
  std::vector<std::vector<uint8_t>> extra;
  CallbackOutput callback_output;
  JxlPixelFormat ec_format = {1, p.format.data_type, p.format.endianness,
                              p.format.align};
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      return result;
    } else if (status == JXL_DEC_SUCCESS) {
      break;
    } else if (status == JXL_DEC_BASIC_INFO) {
      EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec.get(), &info));
      if (p.region.xsize() > 0) {
        EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetImageRegion(
                                       dec.get(), p.region.x0(), p.region.y0(),
                                       p.region.xsize(), p.region.ysize()));
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &p.format, &size));
      JxlFrameHeader header;
      EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetFrameHeader(dec.get(), &header));
      result.xsize = header.layer_info.xsize;
      result.ysize = header.layer_info.ysize;
      size_t stride = Stride(p.format, result.xsize);
      EXPECT_EQ(size, stride * (result.ysize - 1) +
                          result.xsize * p.format.num_channels *
                              BytesPerSample(p.format.data_type));
      pixels.assign(stride * result.ysize, 0);
      if (p.use_callback) {
        callback_output = {
            &pixels, result.xsize, result.ysize, stride,
            p.format.num_channels * BytesPerSample(p.format.data_type)};
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutCallback(dec.get(), &p.format,
                                                CopyPixels, &callback_output));
      } else {
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutBuffer(dec.get(), &p.format,
                                              pixels.data(), pixels.size()));
      }
      extra.clear();
      if (p.extra_channels) {
        size_t first = info.alpha_bits > 0 ? 1 : 0;
        for (size_t ec = first; ec < info.num_extra_channels; ec++) {
          EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderExtraChannelBufferSize(
                                         dec.get(), &ec_format, &size, ec));
          extra.emplace_back(size);
          EXPECT_EQ(JXL_DEC_SUCCESS,
                    JxlDecoderSetExtraChannelBuffer(dec.get(), &ec_format,
                                                    extra.back().data(),
                                                    extra.back().size(), ec));
        }
      }
    } else if (status == JXL_DEC_FRAME_PROGRESSION) {
      if (result.after_dc.empty() && result.frames.empty()) {
        EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderFlushImage(dec.get()));
        result.after_dc = pixels;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      EXPECT_FALSE(callback_output.out_of_bounds);
      result.frames.push_back(pixels);
      result.extra.push_back(extra);
    } else {
      ADD_FAILURE() << "unexpected status " << status;
      return result;
    }
  }
  result.success = true;
  return result;
}

// The `region` of the `xsize` pixels wide image `pixels` of `format`.
std::vector<uint8_t> Crop(const std::vector<uint8_t>& pixels, size_t xsize,
                          const JxlPixelFormat& format, const Rect& region) {
  size_t pixel_size = format.num_channels * BytesPerSample(format.data_type);
  size_t stride = Stride(format, xsize);
  size_t crop_stride = Stride(format, region.xsize());
  std::vector<uint8_t> out(crop_stride * region.ysize(), 0);
  for (size_t y = 0; y < region.ysize(); y++) {
    memcpy(
        out.data() + y * crop_stride,
        pixels.data() + (region.y0() + y) * stride + region.x0() * pixel_size,
        region.xsize() * pixel_size);
  }
  return out;
}

// Decodes `data` whole and with `region` and checks that the region is the
// same, also of the extra channels and of the output after the DC.
void ExpectRegionOfWhole(const std::vector<uint8_t>& data, DecodeParams p,
                         const Rect& region) {
  p.region = Rect();
  Decoded whole = Decode(data, p);
  ASSERT_TRUE(whole.success);
  p.region = region;
  Decoded part = Decode(data, p);
  ASSERT_TRUE(part.success);
  EXPECT_EQ(region.xsize(), part.xsize);
  EXPECT_EQ(region.ysize(), part.ysize);
  ASSERT_EQ(whole.frames.size(), part.frames.size());
  JxlPixelFormat ec_format = {1, p.format.data_type, p.format.endianness,
                              p.format.align};
  for (size_t f = 0; f < whole.frames.size(); f++) {
    EXPECT_TRUE(Crop(whole.frames[f], whole.xsize, p.format, region) ==
                part.frames[f])
        << "frame " << f;
    ASSERT_EQ(whole.extra[f].size(), part.extra[f].size());
    for (size_t ec = 0; ec < whole.extra[f].size(); ec++) {
      std::vector<uint8_t> expected =
          Crop(whole.extra[f][ec], whole.xsize, ec_format, region);
      expected.resize(part.extra[f][ec].size());
      EXPECT_TRUE(expected == part.extra[f][ec])
          << "frame " << f << ", extra channel " << ec;
    }
  }
  // There is an output after the DC with the region if and only if there is one
  // without, the same.
  if (p.flush_after_dc) {
    ASSERT_EQ(whole.after_dc.empty(), part.after_dc.empty());
    if (!whole.after_dc.empty()) {
      EXPECT_TRUE(Crop(whole.after_dc, whole.xsize, p.format, region) ==
                  part.after_dc);
    }
  }
}

TEST(DecodeRegionTest, Orientations) {
  for (uint32_t orientation = 1; orientation <= 8; orientation++) {
    EncodeParams e;
    e.name = "orientation";
    e.xsize = 600;
    e.ysize = 450;
    e.orientation = static_cast<JxlOrientation>(orientation);
    std::vector<uint8_t> data = Encode(e);
    for (bool keep_orientation : {false, true}) {
      for (JxlDataType type : {JXL_TYPE_UINT8, JXL_TYPE_FLOAT}) {
        SCOPED_TRACE(testing::Message()
                     << "orientation " << orientation << ", keep orientation "
                     << keep_orientation << ", type " << type);
        DecodeParams p;
        p.format = {4, type, JXL_NATIVE_ENDIAN, 0};
        p.keep_orientation = keep_orientation;
        // Inside the image in both orientations, across group borders.
        ExpectRegionOfWhole(data, p, Rect(100, 70, 250, 260));
      }
    }
  }
}

TEST(DecodeRegionTest, PixelFormats) {
  EncodeParams e;
  e.name = "formats";
  e.num_channels = 4;
  std::vector<uint8_t> data = Encode(e);
  std::vector<JxlPixelFormat> formats = {
      {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0},
      {4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0},
      {3, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0},
      {4, JXL_TYPE_UINT16, JXL_LITTLE_ENDIAN, 0},
      {4, JXL_TYPE_FLOAT16, JXL_NATIVE_ENDIAN, 0},
      {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0},
      {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 64},
  };
  for (const JxlPixelFormat& format : formats) {
    for (bool use_callback : {false, true}) {
      for (size_t num_threads : {0, 4}) {
        SCOPED_TRACE(testing::Message()
                     << "channels " << format.num_channels << ", type "
                     << format.data_type << ", align " << format.align
                     << ", callback " << use_callback << ", threads "
                     << num_threads);
        DecodeParams p;
        p.format = format;
        p.use_callback = use_callback;
        p.num_threads = num_threads;
        ExpectRegionOfWhole(data, p, Rect(203, 117, 301, 222));
      }
    }
  }
}

TEST(DecodeRegionTest, Positions) {
  EncodeParams e;
  e.name = "positions";
  e.xsize = 700;
  e.ysize = 530;
  std::vector<uint8_t> data = Encode(e);
  std::vector<Rect> regions = {
      Rect(0, 0, 1, 1),         Rect(699, 529, 1, 1), Rect(0, 0, 256, 256),
      Rect(256, 256, 256, 256), Rect(255, 255, 2, 2), Rect(0, 300, 700, 17),
      Rect(511, 0, 3, 530),     Rect(1, 1, 698, 528), Rect(0, 0, 700, 530),
      Rect(600, 400, 100, 130),
  };
  for (const Rect& region : regions) {
    for (size_t num_threads : {0, 3}) {
      SCOPED_TRACE(testing::Message() << "region " << Description(region)
                                      << ", threads " << num_threads);
      DecodeParams p;
      p.num_threads = num_threads;
      ExpectRegionOfWhole(data, p, region);
    }
  }
}

// A case of the kinds of frames, 640 x 480 RGB, lossy with effort 3 unless
// changed.
EncodeParams Case(
    const std::string& name,
    std::vector<std::pair<JxlEncoderFrameSettingId, int64_t>> options = {}) {
  EncodeParams p;
  p.name = name;
  p.options = std::move(options);
  return p;
}

std::vector<EncodeParams> FrameCases() {
  std::vector<EncodeParams> cases;
  cases.push_back(Case("vardct_effort7"));
  cases.back().effort = 7;
  cases.push_back(
      Case("vardct_distance4_epf3", {{JXL_ENC_FRAME_SETTING_EPF, 3}}));
  cases.back().distance = 4.0f;
  cases.push_back(Case(
      "vardct_no_filters",
      {{JXL_ENC_FRAME_SETTING_EPF, 0}, {JXL_ENC_FRAME_SETTING_GABORISH, 0}}));
  cases.push_back(
      Case("progressive_ac", {{JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1},
                              {JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, 1}}));
  cases.push_back(
      Case("progressive_dc", {{JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 1}}));
  cases.push_back(Case("photon_noise"));
  cases.back().float_options = {{JXL_ENC_FRAME_SETTING_PHOTON_NOISE, 3200}};
  cases.push_back(Case("alpha"));
  cases.back().num_channels = 4;
  cases.push_back(Case("extra_channels"));
  cases.back().num_channels = 4;
  cases.back().num_extra = 2;
  cases.push_back(Case("modular_lossless"));
  cases.back().distance = 0;
  cases.push_back(Case("modular_lossless_group128",
                       {{JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, 0}}));
  cases.back().distance = 0;
  cases.back().effort = 2;
  cases.back().num_channels = 4;
  cases.back().num_extra = 1;
  // Squeezed over the whole image.
  cases.push_back(Case("modular_lossy", {{JXL_ENC_FRAME_SETTING_MODULAR, 1}}));
  cases.back().skips = false;
  cases.push_back(
      Case("modular_responsive", {{JXL_ENC_FRAME_SETTING_MODULAR, 1},
                                  {JXL_ENC_FRAME_SETTING_RESPONSIVE, 1}}));
  cases.back().skips = false;
  // Upsampled.
  cases.push_back(Case("resampling2", {{JXL_ENC_FRAME_SETTING_RESAMPLING, 2}}));
  cases.back().skips = false;
  cases.push_back(Case("animation"));
  cases.back().num_frames = 3;
  cases.push_back(Case("animation_blended"));
  cases.back().num_channels = 4;
  cases.back().num_frames = 3;
  cases.back().blend_frames = true;
  return cases;
}

class DecodeRegionFrameTest : public ::testing::TestWithParam<EncodeParams> {};

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(DecodeRegionFrameTestInstantiation,
                                   DecodeRegionFrameTest,
                                   testing::ValuesIn(FrameCases()),
                                   testing::PrintToStringParamName());

TEST_P(DecodeRegionFrameTest, RegionOfWhole) {
  const EncodeParams& e = GetParam();
  std::vector<uint8_t> data = Encode(e);
  for (const Rect& region : {Rect(130, 90, 300, 250), Rect(0, 470, 640, 10)}) {
    for (size_t num_threads : {0, 4}) {
      SCOPED_TRACE(testing::Message() << "region " << Description(region)
                                      << ", threads " << num_threads);
      DecodeParams p;
      p.num_threads = num_threads;
      p.extra_channels = e.num_extra > 0;
      ExpectRegionOfWhole(data, p, region);
    }
  }
}

TEST_P(DecodeRegionFrameTest, AfterDC) {
  const EncodeParams& e = GetParam();
  if (e.distance == 0 || e.num_frames > 0) {
    GTEST_SKIP() << "no output after the DC";
  }
  std::vector<uint8_t> data = Encode(e);
  DecodeParams p;
  p.flush_after_dc = true;
  ExpectRegionOfWhole(data, p, Rect(130, 90, 300, 250));
}

// The byte ranges of the sections of the last frame of the codestream `data`,
// by section id, the dimensions of the frame and its upsampling.
void FrameSections(const std::vector<uint8_t>& data, FrameDimensions* frame_dim,
                   std::vector<std::pair<size_t, size_t>>* sections,
                   size_t* upsampling) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  // The codestream, bare or in a jxlc box.
  size_t base = 0;
  size_t size = data.size();
  if (data.size() < 2 || data[0] != 0xff || data[1] != 0x0a) {
    size_t pos = 0;
    for (;;) {
      ASSERT_LE(pos + 8, data.size());
      size_t box_size = LoadBE32(data.data() + pos);
      if (box_size == 0) box_size = data.size() - pos;
      if (memcmp(data.data() + pos + 4, "jxlc", 4) == 0) {
        base = pos + 8;
        size = box_size - 8;
        break;
      }
      pos += box_size;
    }
  }
  BitReader br(Bytes(data.data() + base, size));
  ASSERT_EQ(br.ReadFixedBits<16>(), 0x0AFFu);
  CodecMetadata metadata;
  ASSERT_TRUE(ReadSizeHeader(&br, &metadata.size));
  ASSERT_TRUE(ReadImageMetadata(&br, &metadata.m));
  metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
  ASSERT_TRUE(Bundle::Read(&br, &metadata.transform_data));
  ASSERT_FALSE(metadata.m.color_encoding.WantICC());
  ASSERT_TRUE(br.JumpToByteBoundary());
  for (;;) {
    FrameHeader frame_header(&metadata);
    ASSERT_TRUE(ReadFrameHeader(&br, &frame_header));
    *frame_dim = frame_header.ToFrameDimensions();
    *upsampling = frame_header.upsampling;
    size_t toc_entries =
        NumTocEntries(frame_dim->num_groups, frame_dim->num_dc_groups,
                      frame_header.passes.num_passes);
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> sizes;
    uint64_t total_size;
    ASSERT_TRUE(ReadGroupOffsets(memory_manager, toc_entries, &br, &offsets,
                                 &sizes, &total_size));
    size_t start = base + br.TotalBitsConsumed() / kBitsPerByte;
    sections->clear();
    for (size_t i = 0; i < toc_entries; i++) {
      sections->emplace_back(start + offsets[i], sizes[i]);
    }
    br.SkipBits(total_size * kBitsPerByte);
    if (frame_header.is_last) break;
  }
  ASSERT_TRUE(br.Close());
}

// Whether `rect` and `region` enlarged by `border` in all directions intersect.
bool Near(const Rect& rect, const Rect& region, size_t border) {
  return rect.x0() < region.x1() + border && region.x0() < rect.x1() + border &&
         rect.y0() < region.y1() + border && region.y0() < rect.y1() + border;
}

// `data` with garbage in the sections of the groups of the last frame that are
// far from `region`, which a decoder of the region need not read; all passes of
// the AC groups farther than 64 pixels and the DC groups farther than a group.
void CorruptFarGroups(const std::vector<uint8_t>& data,
                      const Rect& image_region, std::vector<uint8_t>* corrupted,
                      size_t* num_ac_groups, size_t* num_dc_groups) {
  FrameDimensions frame_dim;
  std::vector<std::pair<size_t, size_t>> sections;
  size_t up;
  ASSERT_NO_FATAL_FAILURE(FrameSections(data, &frame_dim, &sections, &up));
  // The region in pixels of the frame.
  Rect region(image_region.x0() / up, image_region.y0() / up,
              DivCeil(image_region.x1(), up) - image_region.x0() / up,
              DivCeil(image_region.y1(), up) - image_region.y0() / up);
  size_t num_passes =
      (sections.size() - 2 - frame_dim.num_dc_groups) / frame_dim.num_groups;
  ASSERT_EQ(2 + frame_dim.num_dc_groups + num_passes * frame_dim.num_groups,
            sections.size());
  *corrupted = data;
  *num_ac_groups = 0;
  *num_dc_groups = 0;
  Rng rng(1);
  auto corrupt = [&](size_t id) {
    for (size_t i = 0; i < sections[id].second; i++) {
      (*corrupted)[sections[id].first + i] = rng.UniformU(0, 256);
    }
  };
  for (size_t i = 0; i < frame_dim.num_dc_groups; i++) {
    Rect blocks = frame_dim.DCGroupRect(i);
    Rect dc_group(blocks.x0() * kBlockDim, blocks.y0() * kBlockDim,
                  blocks.xsize() * kBlockDim, blocks.ysize() * kBlockDim);
    if (!Near(dc_group, region, frame_dim.group_dim + 64)) {
      corrupt(1 + i);
      ++*num_dc_groups;
    }
  }
  for (size_t g = 0; g < frame_dim.num_groups; g++) {
    if (Near(frame_dim.GroupRect(g), region, 64)) continue;
    for (size_t pass = 0; pass < num_passes; pass++) {
      corrupt(2 + frame_dim.num_dc_groups + pass * frame_dim.num_groups + g);
    }
    ++*num_ac_groups;
  }
}

// Whether the region of `data` is decoded the same with garbage in the groups
// far from it, i.e. whether these are not decoded.
bool SkipsFarGroups(const std::vector<uint8_t>& data, DecodeParams p) {
  std::vector<uint8_t> corrupted;
  size_t num_ac_groups;
  size_t num_dc_groups;
  CorruptFarGroups(data, p.region, &corrupted, &num_ac_groups, &num_dc_groups);
  EXPECT_LT(0u, num_ac_groups);
  Decoded expected = Decode(data, p);
  EXPECT_TRUE(expected.success);
  Decoded part = Decode(corrupted, p);
  return part.success && expected.frames == part.frames &&
         expected.extra == part.extra;
}

TEST_P(DecodeRegionFrameTest, SkipsGroupsOutsideRegion) {
  const EncodeParams& e = GetParam();
  if (e.num_frames > 0) GTEST_SKIP() << "only for still images";
  std::vector<uint8_t> data = Encode(e);
  DecodeParams p;
  p.region = Rect(70, 90, 300, 250);
  p.extra_channels = e.num_extra > 0;
  for (size_t num_threads : {0, 4}) {
    p.num_threads = num_threads;
    EXPECT_EQ(e.skips, SkipsFarGroups(data, p)) << "threads " << num_threads;
  }
}

TEST(DecodeRegionTest, SkipsGroupsOutsideRegion) {
  EncodeParams e;
  e.name = "large";
  e.xsize = 2300;
  e.ysize = 2150;
  std::vector<uint8_t> data = Encode(e);
  FrameDimensions frame_dim;
  std::vector<std::pair<size_t, size_t>> sections;
  size_t upsampling;
  ASSERT_NO_FATAL_FAILURE(
      FrameSections(data, &frame_dim, &sections, &upsampling));
  ASSERT_EQ(1u, upsampling);
  ASSERT_EQ(4u, frame_dim.num_dc_groups);
  ASSERT_EQ(1u + frame_dim.num_dc_groups + 1u + frame_dim.num_groups,
            sections.size());

  for (const Rect& region :
       {Rect(300, 400, 200, 150), Rect(1950, 1900, 200, 200)}) {
    SCOPED_TRACE(testing::Message() << "region " << Description(region));
    std::vector<uint8_t> corrupted;
    size_t num_ac_groups;
    size_t num_dc_groups;
    CorruptFarGroups(data, region, &corrupted, &num_ac_groups, &num_dc_groups);
    EXPECT_LT(frame_dim.num_groups / 2, num_ac_groups);
    // At the corner of the DC groups all are needed.
    if (region.x0() < 1024) EXPECT_LT(0u, num_dc_groups);
    DecodeParams p;
    p.num_threads = 4;
    p.region = region;
    EXPECT_TRUE(SkipsFarGroups(data, p));
    // The whole image cannot be decoded from the garbage.
    p.region = Rect();
    Decoded whole = Decode(corrupted, p);
    Decoded original = Decode(data, p);
    ASSERT_TRUE(original.success);
    EXPECT_TRUE(!whole.success || whole.frames != original.frames);
    // The region near DC group borders is the region of the whole image.
    ExpectRegionOfWhole(data, DecodeParams(), region);
  }
}

TEST(DecodeRegionTest, ApiErrors) {
  EncodeParams e;
  e.name = "api";
  std::vector<uint8_t> data = Encode(e);
  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  // No basic info yet.
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderSetImageRegion(dec.get(), 0, 0, 10, 10));
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSubscribeEvents(
                                 dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                                JXL_DEC_FULL_IMAGE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetInput(dec.get(), data.data(), data.size()));
  JxlDecoderCloseInput(dec.get());
  EXPECT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec.get()));
  // Empty or outside of the image.
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderSetImageRegion(dec.get(), 0, 0, 0, 10));
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderSetImageRegion(dec.get(), 640, 0, 1, 1));
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderSetImageRegion(dec.get(), 600, 0, 41, 1));
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderSetImageRegion(dec.get(), 1, 0, 0xFFFFFFFFu, 1));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageRegion(dec.get(), 600, 400, 40, 80));
  EXPECT_EQ(JXL_DEC_FRAME, JxlDecoderProcessInput(dec.get()));
  // At the frame the region applies to its size.
  JxlFrameHeader header;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetFrameHeader(dec.get(), &header));
  EXPECT_EQ(40u, header.layer_info.xsize);
  EXPECT_EQ(80u, header.layer_info.ysize);
  JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  size_t size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderImageOutBufferSize(dec.get(), &format, &size));
  EXPECT_EQ(40u * 80u * 3u, size);
  // The whole image again.
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetImageRegion(dec.get(), 0, 0, 0, 0));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderImageOutBufferSize(dec.get(), &format, &size));
  EXPECT_EQ(640u * 480u * 3u, size);
  // Not while an output is set.
  std::vector<uint8_t> pixels(size);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels.data(),
                                        pixels.size()));
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderSetImageRegion(dec.get(), 0, 0, 10, 10));
  // Not without coalescing.
  JxlDecoderPtr dec2 = JxlDecoderMake(nullptr);
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetCoalescing(dec2.get(), JXL_FALSE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec2.get(), JXL_DEC_BASIC_INFO));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetInput(dec2.get(), data.data(), data.size()));
  EXPECT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec2.get()));
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderSetImageRegion(dec2.get(), 0, 0, 10, 10));
}

}  // namespace
}  // namespace jxl
