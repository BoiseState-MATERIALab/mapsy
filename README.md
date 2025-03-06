# MapSy
## Mapping the Contact Space Around Atomistic Substrates Using Local Symmetry Information

MapSy is a tool designed to automatically identify inequivalent positions in the space surrounding a substrate and generate local-symmetry-invariant features for machine-learning tasks.

## Goal Applications

1. Automatically identify inequivalent positions in the space surrounding a substrate.
2. Generate local-symmetry-invariant features for machine-learning tasks.

## Features

- **Automatic Identification of Adsorption Sites**: Identify and classify adsorption sites based on local symmetry.
- **Machine Learning Integration**: Generate features that can be used in various machine-learning models.

## Algorithm

1. **Generate the Contact Space (CS)**: Create a grid-data representation of the contact space.
2. **Compute Descriptors for CS**: Calculate features for the contact space.
3. **Feature Selection**: Perform dimensionality reduction on the computed features.
4. **Select N Points (Sites)**: Choose N points from the contact space.
5. **Classify Points**: Classify the selected points according to their relevance (hierarchy).

## Installation

To install the project, you need to have Python 3.8 or higher. You can install the required dependencies using the following command:

```sh 
pip install -r requirements.txt
```

## Usage
To use MapSy, you can run the provided Jupyter notebooks in the docs and examples directories. For example, to run the examples/Pt/planar_maps_ideal.ipynb notebook, use the following command:

```sh
jupyter notebook planar_maps_ideal.ipynb
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the terms of the license found in the LICENSE file.

## Authors
The following people contributed to this project:
* Oliviero Andreussi
* Thiago da Silva
* Jalen Lu
* Zayah Cortright
* Md Sharif Khan
* Denis Mulumba

## Funding Support
This project was partially supported by the NSF CAREER award number 2306929. We also acknowledge the NSF CyberTraining award number 2321102 and, in particular, the 2024 Q-MS Hackathon that have enabled the development of a significant part of the software in the beta release of this package. Jalen Lu and Zayah Cortright acknowledge Dr Warner and the Snake River Local Section Project SEED Site funded by the American Chemical Society. We also acknowledge the Deaprtment of Energy's FAIR program and in particular the reviewers of the 2023 applications for considering the ideas in this work to be standard and not innovative. 

## Contact
For any questions or inquiries, please contact the maintainer:
    Oliviero Andreussi: olivieroandreuss@boisestate.edu
