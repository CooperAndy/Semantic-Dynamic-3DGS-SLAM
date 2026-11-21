#include "TsddMaskFilter.h"

#include <openssl/evp.h>

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ORB_SLAM3
{
namespace
{

struct PendingMaskContext
{
    bool active = false;
    cv::Mat mask;
    TsddMaskPolicy policy = TsddMaskPolicy::O0_NONE;
    TsddMaskStats *stats = nullptr;
};

thread_local PendingMaskContext g_pending_mask_context;

std::string Lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::string Sha256File(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if(!input)
        throw std::runtime_error("TSDD mask file missing: " + path);
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    if(!context)
        throw std::runtime_error("cannot allocate SHA256 context");
    if(EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1)
    {
        EVP_MD_CTX_free(context);
        throw std::runtime_error("cannot initialize SHA256");
    }
    char buffer[1 << 16];
    while(input)
    {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        if(count > 0 && EVP_DigestUpdate(context, buffer, static_cast<std::size_t>(count)) != 1)
        {
            EVP_MD_CTX_free(context);
            throw std::runtime_error("cannot update SHA256");
        }
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if(EVP_DigestFinal_ex(context, digest, &length) != 1)
    {
        EVP_MD_CTX_free(context);
        throw std::runtime_error("cannot finalize SHA256");
    }
    EVP_MD_CTX_free(context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for(unsigned int i = 0; i < length; ++i)
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    return output.str();
}

std::string ParentPath(const std::string &path)
{
    const std::size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

bool IsAbsolutePath(const std::string &path)
{
    return !path.empty() && (path[0] == '/' || path[0] == '\\' || (path.size() > 2 && path[1] == ':'));
}

std::string JoinPath(const std::string &root, const std::string &child)
{
    if(IsAbsolutePath(child))
        throw std::runtime_error("state_map.path must be relative to asset_root");
    if(child.find("..") != std::string::npos)
        throw std::runtime_error("state_map.path may not contain '..'");
    if(root.empty() || root == ".")
        return child;
    const char last = root[root.size() - 1];
    return root + ((last == '/' || last == '\\') ? "" : "/") + child;
}

std::uint32_t ReadLittleEndian(const unsigned char *bytes, std::size_t count)
{
    std::uint32_t value = 0;
    for(std::size_t i = 0; i < count; ++i)
        value |= static_cast<std::uint32_t>(bytes[i]) << (8 * i);
    return value;
}

cv::Mat LoadNpyUint8TwoDimensional(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if(!input)
        throw std::runtime_error("cannot open state map: " + path);
    unsigned char magic[6];
    input.read(reinterpret_cast<char *>(magic), 6);
    const unsigned char expected[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if(!input || !std::equal(magic, magic + 6, expected))
        throw std::runtime_error("state map is not a NumPy .npy file");
    unsigned char version[2];
    input.read(reinterpret_cast<char *>(version), 2);
    if(!input || version[0] < 1 || version[0] > 3)
        throw std::runtime_error("unsupported NumPy version");
    const std::size_t length_bytes = version[0] == 1 ? 2 : 4;
    unsigned char encoded_length[4] = {0, 0, 0, 0};
    input.read(reinterpret_cast<char *>(encoded_length), static_cast<std::streamsize>(length_bytes));
    const std::uint32_t header_length = ReadLittleEndian(encoded_length, length_bytes);
    if(!input || header_length == 0 || header_length > (1u << 20))
        throw std::runtime_error("invalid NumPy header length");
    std::string header(header_length, '\0');
    input.read(&header[0], static_cast<std::streamsize>(header.size()));
    if(!input)
        throw std::runtime_error("truncated NumPy header");
    if(header.find("'fortran_order': False") == std::string::npos &&
       header.find("\"fortran_order\": False") == std::string::npos)
        throw std::runtime_error("Fortran-order state map is unsupported");
    const std::regex descr_pattern("['\\\"]descr['\\\"]\\s*:\\s*['\\\"]([|<>=]?u1)['\\\"]");
    std::smatch descr_match;
    if(!std::regex_search(header, descr_match, descr_pattern))
        throw std::runtime_error("state map dtype must be uint8");
    const std::regex shape_pattern("['\\\"]shape['\\\"]\\s*:\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,?\\s*\\)");
    std::smatch shape_match;
    if(!std::regex_search(header, shape_match, shape_pattern))
        throw std::runtime_error("state map must be two-dimensional");
    const int rows = std::stoi(shape_match[1].str());
    const int cols = std::stoi(shape_match[2].str());
    if(rows <= 0 || cols <= 0)
        throw std::runtime_error("state map shape must be positive");
    cv::Mat result(rows, cols, CV_8UC1);
    const std::size_t required = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    input.read(reinterpret_cast<char *>(result.data), static_cast<std::streamsize>(required));
    if(input.gcount() != static_cast<std::streamsize>(required))
        throw std::runtime_error("truncated state map payload");
    char trailing = 0;
    if(input.read(&trailing, 1))
        throw std::runtime_error("state map has unexpected trailing bytes");
    return result;
}

void ValidateMaskValues(const cv::Mat &mask, TsddMaskPolicy policy)
{
    for(int y = 0; y < mask.rows; ++y)
    {
        const std::uint8_t *row = mask.ptr<std::uint8_t>(y);
        for(int x = 0; x < mask.cols; ++x)
        {
            const std::uint8_t value = row[x];
            if(policy == TsddMaskPolicy::O3_TSDD_DYNAMIC)
            {
                if(value != 0 && value != 1 && value != 2 && value != 255)
                    throw std::runtime_error("O3 state map contains value outside {0,1,2,255}");
            }
            else if(value != 0 && value != 1)
                throw std::runtime_error("O1/O2 binary mask contains value outside {0,1}");
        }
    }
}

bool ShouldRemove(std::uint8_t value, TsddMaskPolicy policy)
{
    if(policy == TsddMaskPolicy::O1_MOVABLE_SEMANTIC || policy == TsddMaskPolicy::O2_BINARY_MOVING)
        return value == 1;
    if(policy == TsddMaskPolicy::O3_TSDD_DYNAMIC)
        return value == 2;
    return false;
}

void ValidateEncoding(const cv::FileNode &root, TsddMaskPolicy policy)
{
    if(policy == TsddMaskPolicy::O3_TSDD_DYNAMIC)
    {
        const cv::FileNode encoding = root["encoding"];
        if(encoding.empty() || static_cast<int>(encoding["Static"]) != 0 ||
           static_cast<int>(encoding["Uncertain"]) != 1 ||
           static_cast<int>(encoding["Dynamic"]) != 2 ||
           static_cast<int>(encoding["Ignore"]) != 255)
            throw std::runtime_error("O3 sidecar encoding must be Static=0, Uncertain=1, Dynamic=2, Ignore=255");
    }
    else
    {
        std::string mask_kind;
        root["mask_kind"] >> mask_kind;
        const std::string expected = policy == TsddMaskPolicy::O1_MOVABLE_SEMANTIC ? "movable_semantic" : "binary_moving";
        if(mask_kind != expected)
            throw std::runtime_error("binary mask sidecar mask_kind mismatch");
    }
}

} // namespace

TsddMaskPolicy ParseTsddMaskPolicy(const std::string &text)
{
    const std::string value = Lower(text);
    if(value == "o0") return TsddMaskPolicy::O0_NONE;
    if(value == "o1") return TsddMaskPolicy::O1_MOVABLE_SEMANTIC;
    if(value == "o2") return TsddMaskPolicy::O2_BINARY_MOVING;
    if(value == "o3") return TsddMaskPolicy::O3_TSDD_DYNAMIC;
    throw std::runtime_error("unknown mask policy: " + text);
}

const char *TsddMaskPolicyName(TsddMaskPolicy policy)
{
    switch(policy)
    {
        case TsddMaskPolicy::O0_NONE: return "O0";
        case TsddMaskPolicy::O1_MOVABLE_SEMANTIC: return "O1";
        case TsddMaskPolicy::O2_BINARY_MOVING: return "O2";
        case TsddMaskPolicy::O3_TSDD_DYNAMIC: return "O3";
    }
    return "INVALID";
}

void SetPendingTsddMaskContext(
    const cv::Mat &left_aligned_mask,
    TsddMaskPolicy policy,
    TsddMaskStats *stats)
{
    if(policy == TsddMaskPolicy::O0_NONE)
        throw std::runtime_error("O0 must use the historical unmasked Frame path");
    if(g_pending_mask_context.active)
        throw std::runtime_error("nested TSDD mask context is not allowed");
    g_pending_mask_context.active = true;
    g_pending_mask_context.mask = left_aligned_mask.clone();
    g_pending_mask_context.policy = policy;
    g_pending_mask_context.stats = stats;
}

void ClearPendingTsddMaskContext()
{
    g_pending_mask_context = PendingMaskContext();
}

void ApplyPendingTsddMaskToStereoFeatures(
    const cv::Size &left_image_size,
    const cv::Size &right_image_size,
    std::vector<cv::KeyPoint> &left_keys,
    cv::Mat &left_descriptors,
    const std::vector<cv::KeyPoint> &right_keys,
    const cv::Mat &right_descriptors)
{
    if(!g_pending_mask_context.active)
        return;
    const cv::Mat mask = g_pending_mask_context.mask;
    const TsddMaskPolicy policy = g_pending_mask_context.policy;
    TsddMaskStats *stats = g_pending_mask_context.stats;
    ClearPendingTsddMaskContext();
    ApplyTsddMaskToStereoFeatures(mask, policy, left_image_size, right_image_size,
                                  left_keys, left_descriptors, right_keys, right_descriptors, stats);
}

LoadedTsddMask LoadTsddMaskSidecar(
    const std::string &sidecar_path,
    const std::string &asset_root,
    const std::string &expected_sequence_id,
    const std::string &expected_frame_id,
    const cv::Size &expected_left_image_size,
    TsddMaskPolicy policy)
{
    if(policy == TsddMaskPolicy::O0_NONE)
        throw std::runtime_error("O0 must not load a mask sidecar");
    cv::FileStorage storage(sidecar_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if(!storage.isOpened())
        throw std::runtime_error("cannot open mask sidecar: " + sidecar_path);
    const cv::FileNode root = storage.root();
    if(static_cast<int>(root["schema_version"]) != 1)
        throw std::runtime_error("mask sidecar schema_version must be 1");
    std::string sequence_id;
    std::string frame_id;
    std::string dtype;
    root["sequence_id"] >> sequence_id;
    root["frame_id"] >> frame_id;
    root["dtype"] >> dtype;
    if(sequence_id != expected_sequence_id)
        throw std::runtime_error("mask sidecar sequence_id mismatch");
    if(frame_id != expected_frame_id)
        throw std::runtime_error("mask sidecar frame_id mismatch");
    if(dtype != "uint8")
        throw std::runtime_error("mask sidecar dtype must be uint8");
    std::vector<int> declared_shape;
    root["shape"] >> declared_shape;
    if(declared_shape.size() != 2 || declared_shape[0] != expected_left_image_size.height || declared_shape[1] != expected_left_image_size.width)
        throw std::runtime_error("mask sidecar shape does not match left image");
    ValidateEncoding(root, policy);
    const cv::FileNode reference = root["state_map"];
    if(reference.empty())
        throw std::runtime_error("mask sidecar state_map is missing");
    std::string relative_path;
    std::string expected_sha;
    std::string map_dtype;
    reference["path"] >> relative_path;
    reference["sha256"] >> expected_sha;
    reference["dtype"] >> map_dtype;
    std::vector<int> map_shape;
    reference["shape"] >> map_shape;
    if(relative_path.empty() || expected_sha.size() != 64 || map_dtype != "uint8" || map_shape != declared_shape)
        throw std::runtime_error("state_map reference is incomplete or inconsistent");
    const std::string root_path = asset_root.empty() ? ParentPath(sidecar_path) : asset_root;
    const std::string map_path = JoinPath(root_path, relative_path);
    const std::string actual_sha = Sha256File(map_path);
    if(Lower(expected_sha) != actual_sha)
        throw std::runtime_error("state_map SHA256 mismatch");
    cv::Mat mask = LoadNpyUint8TwoDimensional(map_path);
    if(mask.type() != CV_8UC1 || mask.size() != expected_left_image_size)
        throw std::runtime_error("actual state_map dtype/shape mismatch");
    ValidateMaskValues(mask, policy);
    LoadedTsddMask result;
    result.mask = mask;
    result.sequence_id = sequence_id;
    result.frame_id = frame_id;
    result.mask_sha256 = actual_sha;
    return result;
}

void ApplyTsddMaskToStereoFeatures(
    const cv::Mat &left_aligned_mask,
    TsddMaskPolicy policy,
    const cv::Size &left_image_size,
    const cv::Size &right_image_size,
    std::vector<cv::KeyPoint> &left_keys,
    cv::Mat &left_descriptors,
    const std::vector<cv::KeyPoint> &right_keys,
    const cv::Mat &right_descriptors,
    TsddMaskStats *stats)
{
    TsddMaskStats local;
    local.original_left = left_keys.size();
    local.original_right = right_keys.size();
    local.kept_right = right_keys.size();
    if(left_image_size != right_image_size)
        throw std::runtime_error("left/right image sizes differ; no implicit mask coordinate mapping is allowed");
    if(!left_descriptors.empty() && left_descriptors.rows != static_cast<int>(left_keys.size()))
        throw std::runtime_error("left keypoint/descriptor count mismatch");
    if(!right_descriptors.empty() && right_descriptors.rows != static_cast<int>(right_keys.size()))
        throw std::runtime_error("right keypoint/descriptor count mismatch");
    if(policy == TsddMaskPolicy::O0_NONE)
    {
        if(!left_aligned_mask.empty())
            throw std::runtime_error("O0 must not receive a mask");
        local.kept_left = left_keys.size();
        if(stats) *stats = local;
        return;
    }
    if(left_aligned_mask.type() != CV_8UC1 || left_aligned_mask.size() != left_image_size)
        throw std::runtime_error("mask must be left-aligned CV_8UC1 with exact image shape");
    ValidateMaskValues(left_aligned_mask, policy);
    std::vector<cv::KeyPoint> kept_keys;
    kept_keys.reserve(left_keys.size());
    cv::Mat kept_descriptors;
    for(std::size_t i = 0; i < left_keys.size(); ++i)
    {
        const int x = cvFloor(left_keys[i].pt.x);
        const int y = cvFloor(left_keys[i].pt.y);
        if(x < 0 || x >= left_image_size.width || y < 0 || y >= left_image_size.height)
            throw std::runtime_error("keypoint lies outside left image");
        const std::uint8_t value = left_aligned_mask.at<std::uint8_t>(y, x);
        if(policy == TsddMaskPolicy::O3_TSDD_DYNAMIC)
        {
            if(value == 0) ++local.static_left;
            else if(value == 1) ++local.uncertain_left;
            else if(value == 2) ++local.dynamic_left;
            else if(value == 255) ++local.ignore_left;
        }
        if(ShouldRemove(value, policy))
        {
            ++local.removed_left;
            continue;
        }
        kept_keys.push_back(left_keys[i]);
        if(!left_descriptors.empty())
            kept_descriptors.push_back(left_descriptors.row(static_cast<int>(i)));
    }
    left_keys.swap(kept_keys);
    if(!left_descriptors.empty())
        left_descriptors = kept_descriptors.clone();
    local.kept_left = left_keys.size();
    if(stats) *stats = local;
}

} // namespace ORB_SLAM3
