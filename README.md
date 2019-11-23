# Plugins to Phy2
These plugins add additional features to Phy2

The repository has been created from phy1-plugins and made compatible with Phy2 by Thomas Hainmueller. 

## Features
* Reclustering. Reclustering with KlustaKwik 2.0 - dependent on a local version of KlustaKwik, which is provided in the zip file for Windows 10) and python package: pandas. To install write “pip install pandas” in the terminal in your phy environment.
* Outlier removal using the Mahalanobis distance. Standard threshold is 16 standard deviations (adjustable threshold).
* K-means clustering. Standard separation into clusters (adjustable number).
* Export shank info for each unit. This is necessary if you want to know which shank a given unit was detected on.

All new features are accessible from the top menu labeled clustering.

## ControllerSettings - Extra columns in ClusterView
ControllerSettings also allows you to adjust the number of spike displayed i FeatureView (increased to 15,000) and WaveformView (standard: 300). I recommend to delete the local .phy folder in your data folder, when adjusting these parameters.

## Installation 
To install, place the content in your plugins directory (~/.phy/), replacing the existing files and plugins folder.

## How to cite
Please use below DOI for citing these plugins.

<a href="https://zenodo.org/badge/latestdoi/126424002"><img src="https://zenodo.org/badge/126424002.svg" alt="DOI"></a>
