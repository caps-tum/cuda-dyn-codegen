using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using System.Globalization;

namespace MatrixToBmp {
    class Program {
        static void Main(string[] args) {
            var matrix = new List<List<float>>();

            foreach (var line in File.ReadAllLines(args[0])) {
                matrix.Add(line.Trim().Split(' ').Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToList());
            }

            var max = matrix.Max(row => row.Max());

            var bmp = new Bitmap(matrix[0].Count, matrix.Count);

            for (var y = 0; y < matrix.Count; ++y) {
                for (var x = 0; x < matrix[0].Count; ++x) {
                    var c = (int)(255 * matrix[y][x] / max);

                    bmp.SetPixel(x, y, Color.FromArgb(c, c, c));
                }
            }

            bmp.Save(args[1]);
        }
    }
}
