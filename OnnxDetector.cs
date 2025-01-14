using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NSFW_Video_Filter
{
    public abstract class OnnxDetector : BaseDetector
    {
        protected InferenceSession _session;
        protected readonly string _inputTensorName;
        protected readonly string _outputTensorName;

        public OnnxDetector(string modelPath, string inputTensorName, string outputTensorName, int resizeWidth, int resizeHeight) : base(resizeWidth, resizeHeight)
        {
            _inputTensorName = inputTensorName;
            _outputTensorName = outputTensorName;
            _session = new InferenceSession(ReadFileChunked(modelPath));
        }

        protected virtual Tensor<float> ImageToTensor(PreprocessedImage image)
        {
            return new DenseTensor<float>(image.PixelData, _shape);
        }

        protected override float[] RunModel(PreprocessedImage image)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputTensorName, ImageToTensor(image)) };

            using (var results = _session.Run(inputs))
            {
                var output = results.FirstOrDefault(r => r.Name == _outputTensorName);
                var value = output?.Value;
                if (output == null || value == null)
                {
                    return Array.Empty<float>();
                }

                if (value is DenseTensor<Float16> float16)
                {
                    return float16.Buffer.Span.ToArray().Select(x => (float)x).ToArray();
                }

                return output.AsEnumerable<float>().ToArray();
            }
        }

        public override void Dispose()
        {
            _session.Dispose();
        }
    }
}