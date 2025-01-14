using FFMpegCore;
using System.Diagnostics;

namespace NSFW_Video_Filter
{
    internal class Program
    {
        static void Main(string[] args)
        {
            using (var detector = new NSFWDetector())
            {
                var outputPath = @"C:\temp\nsfw\frames";

                Directory.CreateDirectory(outputPath);

                if (Directory.GetFiles(outputPath, "*.*", SearchOption.AllDirectories).Length > 0)
                {
                    throw new Exception("Directory not empty");
                }

                int counter = 1;
                ExtractFrames(@"C:\temp\nsfw\I Now Pronounce Chuck Larry.mp4", bytes =>
                {
                    var probability = (int)(detector.CalcNSFWProbability(bytes) * 100);
                    File.WriteAllBytes(Path.Combine(outputPath, $"{counter}_{probability}.jpg"), bytes);
                    counter++;
                });
            }
        }

        protected static void ExtractFrames(string inputVideo, Action<byte[]> action)
        {
            GlobalFFOptions.Configure(options =>
            {
                options.BinaryFolder = Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), "ffmpeg_binaries");
            });

            var result = FFMpegArguments
                      .FromFileInput(inputVideo)
                      .OutputToPipe(new MultiImagePipeSink(action), options => options
                        .ForceFormat("image2pipe")
                        .WithVideoCodec("mjpeg")
                        .WithCustomArgument("-vsync 0")
                        .WithCustomArgument("-q:v 1")
                      )
                      .ProcessSynchronously();
        }
    }
}
