# Hpmoon

Hpmoon is a parallel and distributed multi-objective genetic algorithm to EEG classification. It is an improved version (still beta) of the one presented at https://github.com/efficomp/Hpmoon. The evolutionary procedure corresponds to an island-based model whose subpopulations are distributed among the nodes of HPC clusters. The application is able to parallelize the evaluation of the individuals using all the CPU-GPU devices simultaneously, which provides up to 4 levels of parallelism.

## Requirements

Hpmoon requires a GCC compiler and OpenCL 1.2 compliant CPU-GPU devices. It also depends on the following APIs and libraries:

* [OpenMPI](https://www.open-mpi.org/doc/current/).
* [AMD APP SDK v2.9.1](http://developer.amd.com/wordpress/media/2012/10/AMD_APP_SDK_Release_Notes_Developer2.pdf) or later.
* [Doxygen](https://www.doxygen.nl/index.html) if you want to generate documentation.

## Documentation

Hpmoon is fully documented in its [Github Pages](https://rotty11.github.io/Hpmoon/). In addition, the `Makefile` file contains a rule to generate [Doxygen](https://www.doxygen.nl/index.html) documentation in the `docs/html` folder.

## Usage

The `docs` folder contains the file `user_guide.pdf` with the instructions necessary to use the program. You can also display help by running the program with the `-h` option.

## Funding

This work has been funded by:

* Spanish *Ministerio de Economía y Competitividad* under grants number TIN2012-32039 and TIN2015-67020-P.
* Spanish [*Ministerio de Ciencia, Innovación y Universidades*](https://www.ciencia.gob.es/) under grant number PGC2018-098813-B-C31.
* [*European Regional Development Fund (ERDF)*](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: right">
  <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/mineco.png" height="60">
  <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/miciu.jpg" height="60">
  <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/erdf.png" height="60">
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Hpmoon © 2015 [EFFICOMP](https://efficomp.ugr.es/).
