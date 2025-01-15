using Emgu.CV.Dnn;
using Microsoft.ML.OnnxRuntime.Tensors;
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

    public class BumbleDetector : BaseDetector
    {
        public BumbleDetector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\Bumble\bumble.onnx"), "x:0", "Identity:0", 480, 480)
        {
            _pixelTransformer = x => (x - 128) / 128;
            _shape = new int[] { 1, _resizeHeight * _resizeWidth * 3 };
        }

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, TensorBase modelOutput)
        {
            var result = (DenseTensor<float>)modelOutput;
            return result[0];
        }
    }

    public class MobileNetV2Detector : BaseDetector
    {
        public MobileNetV2Detector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\GantMan\MobileNet_v2\MobileNetV2.onnx"), "self:0", "sequential/prediction/Softmax:0", 224, 224)
        {
            _pixelTransformer = x => x / 255;
            _padToMaintainAspectRatio = false;
        }

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, TensorBase modelOutput)
        {
            //var labels = new string[] { "Drawing", "Hentai", "Neutral", "Porn", "Sexy" }; 
            var result = (DenseTensor<float>)modelOutput;
            return result.Buffer.Span[1] + result.Buffer.Span[3] + result.Buffer.Span[4];
        }
    }

    public class InceptionV3Detector : BaseDetector
    {
        public InceptionV3Detector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\GantMan\Inception_V3\nsfw.299x299.onnx"), "input_1:0", "dense_3/Softmax:0", 299, 299)
        {
            _pixelTransformer = x => x / 255;
            _padToMaintainAspectRatio = false;
        }

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, TensorBase modelOutput)
        {
            //var labels = new string[] { "Drawing", "Hentai", "Neutral", "Porn", "Sexy" }; 
            var result = (DenseTensor<float>)modelOutput;
            return result.Buffer.Span[1] + result.Buffer.Span[3] + result.Buffer.Span[4];
        }
    }

    public class NudeNetDetector : BaseDetector
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

        protected override float ModelOutputToProbability(PreprocessedImage preprocessed, TensorBase modelOutput)
        {
            //var result = PostProcess_Internal(preprocessed, (DenseTensor<float>)modelOutput);
            return 0;
        }

        protected List<Detection> PostProcess_Internal(PreprocessedImage preprocessed, DenseTensor<float> output)
        {
            if (output.Dimensions.Length != 3 || output.Dimensions[0] != 1)
            {
                throw new ArgumentException("Output tensor must have shape [1, rows, cols]");
            }

            int rows = output.Dimensions[1];
            int cols = output.Dimensions[2];

            Span<float> buffer = output.Buffer.ToArray().AsSpan();

            List<float[]> outputs = new List<float[]>(rows);
            for (int i = 0; i < rows; i++)
            {
                int offset = i * cols;
                float[] row = buffer.Slice(offset, cols).ToArray();
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

                    x -= w / 2;
                    y -= h / 2;

                    x *= (preprocessed.OriginalWidth + preprocessed.XPadding) / (float)preprocessed.Image.Width;
                    y *= (preprocessed.OriginalHeight + preprocessed.YPadding) / (float)preprocessed.Image.Height;
                    w *= (preprocessed.OriginalWidth + preprocessed.XPadding) / (float)preprocessed.Image.Width;
                    h *= (preprocessed.OriginalHeight + preprocessed.YPadding) / (float)preprocessed.Image.Height;

                    x = Math.Max(0, Math.Min(x, preprocessed.OriginalWidth));
                    y = Math.Max(0, Math.Min(y, preprocessed.OriginalHeight));
                    w = Math.Min(w, preprocessed.OriginalWidth - x);
                    h = Math.Min(h, preprocessed.OriginalHeight - y);

                    classIds.Add(classId);
                    scores.Add(maxScore);
                    boxes.Add(new Rectangle((int)x, (int)y, (int)w, (int)h));
                }
            }

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
