#!/bin/sh

cd "${0%/*}"

convert *.tiff -set filename: "%t" %[filename:].png;
rm *.tiff*;
