# Syllable colors
syl_colors = {1:'#000000', # s1
              2:'#B67FED', # s2
              3:'#C06E1B', # s3
              4:'#128080', # s4
              5:'#804912', # s5
              6:'#286DB6', # s6
              7:'#D3D3D3', # s7
              8:'#D3D3D3', # motif_silence
              9:'#349946', # intra-motif note: green #349946
              10:'#FEB4D9', # intro note
              11:'#FFFF6D', # calls
              12:'#00FFFF', # Unlabeled
              13:'#EBEBEB', # silence
              14:'#EBEBEB'} 


# Style dict for Variance Explained and Latent Dispersion figures
style_dict = {
    'color': {
        'default': 'black',
        'ra': '#dd8350',
        'hvc': '#154c79',
        'control_ra': 'k'
    },
    'alpha': {
        'default': 1,
        'all': 1,
        'sua': 0.5
    },
    'linestyle': {
        'default': '-',
        'time': 'dotted',
        'neurons': '--',
        'neuron_control': ':',        
    },
    'barstyle': {
        'default': '',
        'time': '-',
        'neurons': '..',
        'all_neuron_control': '-',
        'sua_neuron_control': '..',
    }
}