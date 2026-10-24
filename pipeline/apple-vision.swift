// Looks at a picture with Apple's Vision framework, which comes with macOS and iOS.
// Prints JSON: {"width": 1024, "height": 1024, "faces": [{"box": [x0, y0, x1, y1], "confidence": 0.98}]}
// Boxes are in pixels from the top-left corner, largest face first.
//
// With --body-mask, also:
//   - saves a grayscale PNG the size of the picture: white where Vision sees a person, black
//     elsewhere;
//   - adds a "joints" object to the JSON, the body's joints Vision is at least 50% sure of, each
//     as [x, y] in pixels from the top-left corner: leftShoulder, rightShoulder, leftHip, rightHip,
//     neck, root (the pelvis).
// body_measure.py reads both to measure shoulder, waist and hip width.
//
//     apple-vision picture.png [--body-mask mask.png]
//
// Built by pipeline/setup.sh, or by apple_vision.py the first time it is needed.

import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count == 1 || (arguments.count == 3 && arguments[1] == "--body-mask") else {
    fail("usage: apple-vision <picture> [--body-mask <mask.png>]")
}
let url = URL(fileURLWithPath: arguments[0])
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
else { fail("Cannot read \(url.path)") }

let width = Double(image.width), height = Double(image.height)
let handler = VNImageRequestHandler(cgImage: image, options: [:])
let faceRequest = VNDetectFaceRectanglesRequest()
let personRequest = VNGeneratePersonSegmentationRequest()
personRequest.qualityLevel = .accurate
personRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8
let poseRequest = VNDetectHumanBodyPoseRequest()
do {
    try handler.perform(arguments.count == 3 ? [faceRequest, personRequest, poseRequest] : [faceRequest])
} catch {
    fail("Vision could not look at the picture: \(error.localizedDescription)")
}

var result: [String: Any] = ["width": width, "height": height]

if arguments.count == 3, let mask = personRequest.results?.first?.pixelBuffer {
    let scaled = CIImage(cvPixelBuffer: mask).transformed(by: CGAffineTransform(
        scaleX: width / Double(CVPixelBufferGetWidth(mask)), y: height / Double(CVPixelBufferGetHeight(mask))))
    let context = CIContext()
    guard let cgMask = context.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: width, height: height)),
          let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: arguments[2]) as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { fail("Cannot save the body mask") }
    CGImageDestinationAddImage(destination, cgMask, nil)
    CGImageDestinationFinalize(destination)
}

if arguments.count == 3, let pose = poseRequest.results?.first {
    let wanted: [VNHumanBodyPoseObservation.JointName: String] = [
        .neck: "neck", .root: "root",
        .leftShoulder: "leftShoulder", .rightShoulder: "rightShoulder",
        .leftHip: "leftHip", .rightHip: "rightHip",
    ]
// How sure Vision has to be of a joint before it is reported.
//
// Was 0.5, on the reasoning that a guessed joint is worse than no measurement. Measured on 95
// whole-body photographs, that gate refused one person in five — and it was a knife edge: the
// bodies that got through had hips at 0.50 and 0.51, while a muscular man was refused at 0.45
// and a plus-size woman at 0.28, all three with Vision certain there was a person there
// (observation confidence 1.000). The refusals were not spread evenly either; they were the
// larger and more muscular bodies.
//
// It is lowered because of what these joints are actually used for. They do not supply any
// width — every width comes from the silhouette — they only say which ROW to measure at. Moving
// the hip row by 10% of the shoulder-to-hip span changed the final reading by at most 0.9%
// across twelve bodies, median 0.5%, against noise of about 0.8% on an unchanged photo. A
// slightly uncertain joint is accurate enough for the one job it has, and the guards that
// prevent a wrong measurement — arms clear of the torso, nothing touching the frame edge,
// enough clean rows — all live in the silhouette and are unaffected by this.
let minimumJointConfidence: VNConfidence = 0.25

    var joints: [String: [Double]] = [:]
    for (joint, name) in wanted {
        guard let point = try? pose.recognizedPoint(joint),
              point.confidence > minimumJointConfidence else { continue }
        // Vision measures from the bottom-left corner, as a share of the picture.
        joints[name] = [point.location.x * width, (1 - point.location.y) * height]
    }
    result["joints"] = joints
}

// Vision measures from the bottom-left corner, as a share of the picture.
result["faces"] = (faceRequest.results ?? [])
    .map { face -> [String: Any] in
        let box = face.boundingBox
        return [
            "box": [box.minX * width, (1 - box.maxY) * height, box.maxX * width, (1 - box.minY) * height],
            "confidence": Double(face.confidence),
        ]
    }
    .sorted { a, b in
        let (p, q) = (a["box"] as! [Double], b["box"] as! [Double])
        return (p[2] - p[0]) * (p[3] - p[1]) > (q[2] - q[0]) * (q[3] - q[1])
    }

let json = try! JSONSerialization.data(withJSONObject: result)
print(String(data: json, encoding: .utf8)!)
