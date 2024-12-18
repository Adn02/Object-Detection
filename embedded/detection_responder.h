/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Provides an interface to take an action based on the output from the person
// detection model.

#ifndef TENSORFLOW_LITE_PLANE_DETECTION_DETECTION_RESPONDER_H_
#define TENSORFLOW_LITE_PLANE_DETECTION_DETECTION_RESPONDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Called every time the results of a plane detection run are available. The
// `plane_score` has the numerical confidence that the captured image contains
// a plane, and `no_plane_score` has the numerical confidence that the image
// does not contain a plane. Typically if plane_score > no_plane_score, the
// image is considered to contain a plane.  This threshold may be adjusted for
// particular applications.
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t plane_score, int8_t no_plane_score);

#endif  // TENSORFLOW_LITE_PLANE_DETECTION_DETECTION_RESPONDER_H_
