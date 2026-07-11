grid_refine_input_section — grid refine functional test fixture (section 0690)

Source (production testdata):
  D:\nornir-testdata\PlatformRaw\IDOC\RC2_4Square_Assembled\TEM\0690\TEM\

Golden C++ output (optional, for target-point parity):
  Grid_Cel96_Mes8_sp4_Mes8_Thr0.5.mosaic

Required inputs:
  Translated_Prune_Max0.5.mosaic
  Leveled/TilePyramid/004/000.png … 003.png

Refine parameters (match golden filename):
  cell=96, mesh=8x8, sp=4, threshold=0.5

Production install:
  {TESTINPUTPATH}/PlatformRaw/IDOC/RC2_4Square_Assembled/TEM/0690/TEM/

C++ reference implementation:
  D:\src\SVN\SCI\trunk (ir-refine-grid)
