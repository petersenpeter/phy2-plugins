# You can also put your plugins in ~/.phy/plugins/.

from phy import IPlugin

try:
    import phycontrib
except:
    pass

c = get_config()

# Plugin directories
c.Plugins.dirs = [r'~/.phy/plugins/']

# Configure GUI plugins
c.TemplateGUI.plugins = [
    'ReclusterMingze',
    'RawDataFilterPlugin',
    'CustomActionPlugin',
    'ImprovedISIAnalysisMingze',
    'GoodLabelsPlugin',
    'StableMahalanobisDetectionMingze'
]