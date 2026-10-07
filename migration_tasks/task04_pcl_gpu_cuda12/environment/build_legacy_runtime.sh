#!/bin/bash
# Builds the original PCL GPU stack with CUDA 11.8 (the last toolkit with texture references) for
# compute_86, so that it runs on sm_120 through driver PTX JIT. Then builds the consumers against it and
# collects them with every non-glibc shared library they load into $OUT (bin/, lib/).
#   build_legacy_runtime.sh PCL_SRC CONSUMERS_DIR OUT
set -euo pipefail
SRC=$1; CONS=$2; OUT=$3
B=$(mktemp -d); I=$B/install
cmake -S "$SRC" -B $B/pcl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$I -DWITH_CUDA=ON -DBUILD_GPU=ON \
  -DBUILD_gpu_kinfu=ON -DBUILD_gpu_kinfu_large_scale=ON -DBUILD_gpu_people=ON -DBUILD_gpu_kinfu_tools=OFF \
  -DBUILD_gpu_kinfu_large_scale_tools=OFF -DBUILD_gpu_people_tools=OFF -DBUILD_apps=OFF -DBUILD_examples=OFF \
  -DBUILD_tools=OFF -DBUILD_global_tests=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF -DWITH_QT=OFF -DWITH_PCAP=OFF \
  -DWITH_LIBUSB=OFF -DWITH_ENSENSO=OFF -DWITH_DAVIDSDK=OFF -DWITH_DSSDK=OFF -DWITH_RSSDK=OFF -DWITH_RSSDK2=OFF \
  -DCUDA_ARCH_BIN=86 > $B/cmake.log
cmake --build $B/pcl > $B/build.log
cmake --install $B/pcl > /dev/null
for r in kinfu_replay people_eval; do
  cmake -S "$CONS/$r" -B $B/$r -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$I > /dev/null
  cmake --build $B/$r -j > /dev/null
done
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/lib"
cp $B/kinfu_replay/kinfu_replay $B/people_eval/people_eval $B/people_eval/ncv_check "$OUT/bin/"
export LD_LIBRARY_PATH=$I/lib:/usr/local/cuda/lib64
for b in "$OUT"/bin/*; do ldd $b; done | awk '/=>/ {print $3}' | sort -u | while read -r l; do
  case $(basename "$l") in
    libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libstdc++.so*|libgcc_s.so*) ;;
    *) cp -L "$l" "$OUT/lib/" ;;
  esac
done
strip --strip-unneeded "$OUT"/lib/libpcl_* "$OUT"/bin/*
rm -rf $B
echo "legacy runtime: $(du -sh "$OUT" | cut -f1)"
