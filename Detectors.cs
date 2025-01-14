using Emgu.CV.Dnn;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Rectangle = System.Drawing.Rectangle;

namespace NSFW_Video_Filter
{
    public class NSFWDetector : IDisposable
    {
        Dictionary<BaseDetector, float> _weightPerDetector;
        public NSFWDetector()
        {
            //todo: check if any need Red & Blue swapped
            _weightPerDetector = new Dictionary<BaseDetector, float>()
            {
                //{ new BumbleDetector(), .1f },
                { new MobileNetV2Detector(), .3f },
                { new InceptionV3Detector(), .5f },
                //{ new NudeNetDetector(), .5f },
            };

            var total = _weightPerDetector.Select(x => x.Value).Sum();
            _weightPerDetector = _weightPerDetector.ToDictionary(x => x.Key, x => x.Value / total);
        }

        public float CalcNSFWProbability(byte[] imageBytes)
        {
            using (var memoryStream = new MemoryStream(imageBytes))
            using (var image = Image.Load<Rgba32>(memoryStream))
            {
                var result = _weightPerDetector.Select(x => new { score = (int)Math.Clamp(x.Key.CalcNSFWProbability(image) * 100, 0, 100), weight = x.Value }).ToList();
                var weightedAverage = result.Select(x => x.score * x.weight).Sum();
                var average = result.Select(x => x.score).Average();
                var standardDeviation = result.Select(x => (float)Math.Pow(x.score - average, 2)).Average();
                if (standardDeviation != 0)
                {
                    var bonus = Math.Clamp(1 + (10 / standardDeviation), 1, 2);
                    weightedAverage *= bonus;
                }
                return Math.Clamp(weightedAverage / 100, 0, 1);
            }
        }

        public void Dispose()
        {
            foreach (var detector in _weightPerDetector.Keys)
            {
                detector.Dispose();
            }
        }
    }

    public class BumbleDetector : OnnxDetector
    {
        public BumbleDetector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\Bumble\bumble.onnx"), "x:0", "Identity:0", 480, 480)
        {
            _pixelTransformer = x => (x - 128) / 128;
            _shape = new int[] { 1, _resizeHeight * _resizeWidth * 3 };
        }

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, float[] modelOutput)
        {
            return modelOutput[0];
        }
    }

    public class MobileNetV2Detector : OnnxDetector
    {
        public MobileNetV2Detector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\GantMan\MobileNet_v2\MobileNetV2.onnx"), "self:0", "sequential/prediction/Softmax:0", 224, 224)
        {
            _pixelTransformer = x => x / 255;
            _padToMaintainAspectRatio = false;
        }

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, float[] modelOutput)
        {
            //var labels = new string[] { "Drawing", "Hentai", "Neutral", "Porn", "Sexy" }; 
            return modelOutput[1] + modelOutput[3] + modelOutput[4];
        }
    }

    public class InceptionV3Detector : OnnxDetector
    {
        public InceptionV3Detector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\GantMan\Inception_V3\nsfw.299x299.onnx"), "input_1:0", "dense_3/Softmax:0", 299, 299)
        {
            _pixelTransformer = x => x / 255;
            _padToMaintainAspectRatio = false;
        }

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, float[] modelOutput)
        {
            //var labels = new string[] { "Drawing", "Hentai", "Neutral", "Porn", "Sexy" }; 
            return modelOutput[1] + modelOutput[3] + modelOutput[4];
        }
    }

    public class NudeNetDetector : OnnxDetector
    {
        public NudeNetDetector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\NudeNet\640m.onnx"), "images", "output0", 640, 640)
        {
            _shape = new int[] { 1, 3, _resizeHeight, _resizeWidth };
            _pixelTransformer = x => x / 255;
        }

        public class Detection
        {
            public string Class { get; set; }
            public float Score { get; set; }
            public int[] Box { get; set; }
        }

        protected readonly string[] _labels = new string[] { "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED" };

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, float[] modelOutput)
        {
            //todo: fix this. modelOuput array length doesn't match what's expected
            return 0;
            /*
            var result = PostProcess_Internal(modelOutput, preprocessed.XPadding, preprocessed.YPadding, preprocessed.RatioX, preprocessed.RatioY, preprocessed.OriginalWidth, preprocessed.OriginalHeight, _resizeWidth, _resizeHeight);
            return new float[0];
            */
        }

        protected List<Detection> PostProcess_Internal(
               float[,,] output,
               int xPad,
               int yPad,
               float xRatio,
               float yRatio,
               int imageOriginalWidth,
               int imageOriginalHeight,
               int modelWidth,
               int modelHeight)
        {
            // Squeeze the output and transpose
            // Assuming output shape is [1, rows, cols], transpose to [cols, rows]
            int rows = output.GetLength(1);
            int cols = output.GetLength(2);

            // Convert the 3D output array to a 2D list for easier processing
            List<float[]> outputs = new List<float[]>();
            for (int i = 0; i < rows; i++)
            {
                float[] row = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    row[j] = output[0, i, j];
                }
                outputs.Add(row);
            }

            List<int> classIds = new List<int>();
            List<float> scores = new List<float>();
            List<Rectangle> boxes = new List<Rectangle>();

            for (int i = 0; i < outputs.Count; i++)
            {
                float[] classesScores = outputs[i].Skip(4).ToArray();
                float maxScore = classesScores.Max();

                if (maxScore >= 0.2f)
                {
                    int classId = Array.IndexOf(classesScores, maxScore);
                    float x = outputs[i][0];
                    float y = outputs[i][1];
                    float w = outputs[i][2];
                    float h = outputs[i][3];

                    // Convert from center coordinates to top-left corner coordinates
                    x -= w / 2;
                    y -= h / 2;

                    // Scale coordinates to original image size
                    x *= (imageOriginalWidth + xPad) / (float)modelWidth;
                    y *= (imageOriginalHeight + yPad) / (float)modelHeight;
                    w *= (imageOriginalWidth + xPad) / (float)modelWidth;
                    h *= (imageOriginalHeight + yPad) / (float)modelHeight;

                    // Clip coordinates to image boundaries
                    x = Math.Max(0, Math.Min(x, imageOriginalWidth));
                    y = Math.Max(0, Math.Min(y, imageOriginalHeight));
                    w = Math.Min(w, imageOriginalWidth - x);
                    h = Math.Min(h, imageOriginalHeight - y);

                    classIds.Add(classId);
                    scores.Add(maxScore);
                    boxes.Add(new Rectangle((int)x, (int)y, (int)w, (int)h));
                }
            }

            // Perform Non-Max Suppression
            var indices = DnnInvoke.NMSBoxes(
                boxes.Select(b => new Rectangle(b.X, b.Y, b.Width, b.Height)).ToArray(),
                scores.ToArray(),
                0.25f,
                0.45f
            );

            List<Detection> detections = new List<Detection>();
            foreach (var i in indices)
            {
                var box = boxes[i];
                float score = scores[i];
                int classId = classIds[i];

                detections.Add(new Detection
                {
                    Class = _labels[classId],
                    Score = score,
                    Box = new int[] { box.X, box.Y, box.Width, box.Height }
                });
            }

            return detections;
        }
    }
}
