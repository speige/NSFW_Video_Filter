using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace NSFW_Video_Filter
{
    public abstract class BaseDetector
    {
        protected InferenceSession _session;
        protected readonly string _inputTensorName;
        protected readonly string _outputTensorName;
        protected readonly int _resizeWidth;
        protected readonly int _resizeHeight;
        protected Func<float, float> _pixelTransformer { get; init; }
        protected int[] _shape;
        protected bool _padToMaintainAspectRatio = true;


        public BaseDetector(string modelPath, string inputTensorName, string outputTensorName, int resizeWidth, int resizeHeight)
        {
            _inputTensorName = inputTensorName;
            _outputTensorName = outputTensorName;
            _resizeWidth = resizeWidth;
            _resizeHeight = resizeHeight;

            _shape = new int[] { 1, _resizeHeight, _resizeWidth, 3 };

            var sessionOptions = new SessionOptions()
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                //LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
            };
            for (var deviceId = 0; deviceId <= 2; deviceId++)
            {
                try
                {
                    sessionOptions.AppendExecutionProvider_DML(deviceId);
                    break;
                }
                catch (Exception e)
                {
                    if (e.Message.Contains("handle is invalid", StringComparison.InvariantCultureIgnoreCase))
                    {
                        break;
                    }

                    Console.WriteLine("Unable to use GPU Acceleration: " + e.Message);
                }
            }
            sessionOptions.AppendExecutionProvider_CPU();

            _session = new InferenceSession(ReadFileChunked(modelPath), sessionOptions);
        }

        public static byte[] ReadFileChunked(string modelPath)
        {
            List<byte> result = new List<byte>();
            var i = 0;
            while (true)
            {
                var fileName = $"{modelPath}.{i}";
                if (!File.Exists(fileName))
                {
                    break;
                }

                result.AddRange(File.ReadAllBytes(fileName));
                i++;
            }

            return result.ToArray();
        }

        public static void SplitFileIntoChunks(string modelPath, int chunkSizeInMB = 50)
        {
            var chunks = File.ReadAllBytes(modelPath).Chunk(chunkSizeInMB * 1024 * 1024).ToList();
            for (var i = 0; i < chunks.Count; i++)
            {
                File.WriteAllBytes($"{modelPath}.{i}", chunks[i]);
            }
        }

        protected class PreprocessedImage : IDisposable
        {
            public Image<Rgba32> Image;
            public float[] PixelData;
            public float RatioX;
            public float RatioY;
            public int XPadding;
            public int YPadding;
            public int OriginalWidth;
            public int OriginalHeight;

            public void Dispose()
            {
                if (Image != null)
                {
                    Image.Dispose();
                }
            }
        }

        protected PreprocessedImage PreprocessImage(Image<Rgba32> image)
        {
            var result = new PreprocessedImage();
            result.Image = image.Clone();
            result.OriginalWidth = result.Image.Width;
            result.OriginalHeight = result.Image.Height;
            result.RatioX = (float)_resizeWidth / result.Image.Width;
            result.RatioY = (float)_resizeHeight / result.Image.Height;
            result.Image.Mutate(ctx => ctx.Resize(new ResizeOptions() { Mode = _padToMaintainAspectRatio ? ResizeMode.Pad : ResizeMode.Stretch, Size = new Size(_resizeWidth, _resizeHeight) }));
            result.PixelData = GetPixelData(result);
            return result;
        }

        protected float[] GetPixelData(PreprocessedImage preprocessed)
        {
            float[] pixelData = new float[preprocessed.Image.Width * preprocessed.Image.Height * 3];
            preprocessed.Image.ProcessPixelRows(pixels => {
                var idx = 0;
                for (int y = 0; y < preprocessed.Image.Height; y++)
                {
                    var rowSpan = pixels.GetRowSpan(y);
                    for (int x = 0; x < preprocessed.Image.Width; x++)
                    {
                        Rgba32 c = rowSpan[x];

                        float red = c.R;
                        float green = c.G;
                        float blue = c.B;

                        if (_pixelTransformer != null)
                        {
                            red = _pixelTransformer(red);
                            green = _pixelTransformer(green);
                            blue = _pixelTransformer(blue);
                        }

                        pixelData[idx + 0] = red;
                        pixelData[idx + 1] = green;
                        pixelData[idx + 2] = blue;

                        idx += 3;
                    }
                }
            });

            return pixelData;
        }

        protected virtual Tensor<float> ImageToTensor(PreprocessedImage image)
        {
            return new DenseTensor<float>(image.PixelData, _shape);
        }

        protected TensorBase RunModel<T>(Tensor<T> input, string inputTensorName, string outputTensorName)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputTensorName, input) };

            using (var results = _session.Run(inputs))
            {
                var output = results.FirstOrDefault(r => r.Name == outputTensorName);
                return output?.Value as TensorBase;
            }
        }

        protected virtual TensorBase RunModel(PreprocessedImage image)
        {
            return RunModel(ImageToTensor(image), _inputTensorName, _outputTensorName);
        }

        public void Dispose()
        {
            _session.Dispose();
        }

        public virtual float CalcNSFWProbability(byte[] imageBytes)
        {
            using (var memoryStream = new MemoryStream(imageBytes))
            using (var image = Image.Load<Rgba32>(memoryStream))
            {
                return CalcNSFWProbability(imageBytes);
            }
        }

        protected abstract float ModelOutputToProbability(PreprocessedImage preprocessed, TensorBase modelOutput);

        public virtual float CalcNSFWProbability(Image<Rgba32> image)
        {
            using (var preprocessed = PreprocessImage(image))
            {
                var output = RunModel(preprocessed);
                return (float)Math.Round(ModelOutputToProbability(preprocessed, output), 2);
            }
        }
    }
}