#include <opencv2/highgui.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include <vector>
#include <mutex>
#include <thread>
#include <memory>

#define THREAD_COUNT std::thread::hardware_concurrency()

using namespace cv;
using namespace cv::img_hash;
namespace fs = std::filesystem;

std::vector<fs::path> allPhotos(const fs::path& path) {
    std::vector<fs::path> files;
    for(const auto& file : fs::recursive_directory_iterator(path)) {
        if (file.is_regular_file())
            files.push_back(file);
    }

    return files;
}

template <typename T>
std::vector<std::vector<T>> splitArray(const std::vector<T>& arr, int n) {
    int avg = arr.size() / n;
    int remainder = arr.size() % n;

    std::vector<std::vector<T>> result(n);
    int start = 0;

    for (int i = 0; i < n; ++i) {
        int end = start + avg + (i < remainder ? 1 : 0);
        result[i] = std::vector<T>(arr.begin() + start, arr.begin() + end);
        start = end;
    }

    return result;
}

template <typename Func, typename T>
void splitWorkOnThread(const Func& f, const std::vector<T>& vec) {
    std::vector<std::thread> threads;
    for(const auto& sub : vec) {
        threads.emplace_back(f, sub);
    }

    for(auto& thread : threads) {
        thread.join();
    }
}

std::string CalcImageHash(const std::string& fileName) {
    Mat image = imread(fileName, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error loading image: " << fileName << std::endl;
        return "";
    }

    Mat resized;
    resize(image, resized, Size(48, 48), INTER_AREA);

    Mat gray_image;
    cvtColor(resized, gray_image, COLOR_BGR2GRAY);

    double avg = mean(gray_image)[0];

    Mat threshold_image;
    threshold(gray_image, threshold_image, avg, 255, THRESH_BINARY);

    std::string hash = "";
    for (int x = 0; x < 48; ++x) {
        for (int y = 0; y < 48; ++y) {
            if (threshold_image.at<uchar>(x, y) == 255) {
                hash += "1";
            } else {
                hash += "0";
            }
        }
    }

    return hash;
}

int CompareHash(const std::string& hash1, const std::string& hash2) {
    int count = 0;
    for (size_t i = 0; i < hash1.size(); ++i) {
        if (hash1[i] != hash2[i]) {
            ++count;
        }
    }
    return count;
}

struct ImageData {
    fs::path path;
    std::string hash;
};

struct TmpData {
    size_t index1;
    size_t index2;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return -1;
    }

    std::vector<fs::path> photos = allPhotos(argv[1]);
    if (photos.empty()) {
        std::cerr << "No images found in the directory." << std::endl;
        return -1;
    }

    std::mutex mutex;
    std::vector<std::shared_ptr<ImageData>> images;

    std::vector<std::vector<fs::path>> sub = splitArray(photos, THREAD_COUNT);
    auto LamdaFun = [&](const std::vector<fs::path> &vec) {
        for (const auto& photo : vec) {
            std::string hash = CalcImageHash(photo.string());
            if (hash.empty()) {
                continue;
            }

            std::shared_ptr<ImageData> data = std::make_shared<ImageData>();
            data->path = photo;
            data->hash = hash;
            
            std::lock_guard<std::mutex> lock(mutex);
            images.push_back(data);
        }
    };
    splitWorkOnThread(LamdaFun, sub);

    std::vector<TmpData> tmp;
    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t j = i + 1; j < images.size(); ++j) {
            tmp.push_back({i, j});
        }
    }

    std::vector<std::vector<TmpData>> sub2 = splitArray(tmp, THREAD_COUNT);
    auto LamdaFun2 = [&](const std::vector<TmpData> &vec) {
        for (const auto& data : vec) {
            int difference = CompareHash(images[data.index1]->hash, images[data.index2]->hash);
            if (difference < 50) {
                std::lock_guard<std::mutex> lock(mutex);
                std::cout << images[data.index1]->path << " | " << images[data.index2]->path << ": similar (difference = " << difference << ")" << std::endl;
            }
        }
    };
    splitWorkOnThread(LamdaFun2, sub2);

    return 0;
}