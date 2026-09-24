#!/usr/bin/env bash
# Tải và giải nén bộ dữ liệu IWSLT15 English-Vietnamese vào data/raw/.
# Môi trường này không có wget nên dùng curl -L thay thế.
set -e

cd "$(dirname "$0")/../data/raw"

curl -L -o train-en-vi.tgz https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz
curl -L -o dev-2012-en-vi.tgz https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz
curl -L -o test-2013-en-vi.tgz https://github.com/stefan-it/nmt-en-vi/raw/master/data/test-2013-en-vi.tgz

tar -xzf train-en-vi.tgz
tar -xzf dev-2012-en-vi.tgz
tar -xzf test-2013-en-vi.tgz
