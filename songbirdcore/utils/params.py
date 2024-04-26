from dataclasses import dataclass


@dataclass
class GlobalParams:
    """Parameters for neural data processing 
    
    Args:
        refractory_period (int): 
        ISI_violation_percentage (int):

    """
    
    # Neural clusters params
    refractory_period = 2 
    ISI_violation_percentage = 3
    
    # GPFA params
    gpfa_max_iter = 2000
    
    

@dataclass
class BirdSpecificParams:
    """Parameters for neural data processing 
    
    Args:
        b1_len_motif:
        b1_syllable_mapping (dict): 
        
        b2_len_motif:
        b2_syllable_mapping (dict): 
        
        b3_len_motif:
        b3_syllable_mapping (dict): 
        
        b4_len_motif:
        b4_syllable_mapping (dict): 
        
    """
    
    data = {
            'z_w12m7_20': {'len_motif': 0.5, 
                           'syllable_mapping': 
                                       {'1': 1, # s1
                                        '2': 2, # s2
                                        '3': 3, # s3
                                        '4': 4, # s4
                                        '5': 5, # s5
                                        '6': 9, # s6 intra-motif note: green #349946
                                        '7': 7, # s7
                                        'i': 10, # 
                                        'I': 10, # intro note
                                        'C': 11, # calls
                                        'u': 12, # Unlabeled
                                        'B': 8, # bout_silence
                                        'S': 13 # silence
                                       }, 
                          },
        
            'z_r12r13_21': {'len_motif': 0.7, 
                           'syllable_mapping': 
                                       {'1': 1, # s1
                                        '2': 2, # s2
                                        '3': 3, # s3
                                        '4': 4, # s4
                                        '5': 5, # s5
                                        '6': 6, # s6
                                        '7': 9, # s7 intra-motif note: green #349946
                                        'i': 10, # 
                                        'I': 10, # intro note
                                        'C': 11, # calls
                                        'u': 12, # Unlabeled
                                        'B': 8, # bout_silence
                                        'S': 13 # silence
                                       }, 
                           },

            'z_c5o30_23': {'len_motif': 0.8, 
                           'syllable_mapping': 
                                       {'1': 1, # s1
                                        '2': 2, # s2
                                        '3': 3, # s3
                                        '4': 4, # s4
                                        '5': 5, # s5
                                        '6': 6, # s6
                                        '7': 7, # s7
                                        'i': 10, # 
                                        'I': 10, # intro note
                                        'C': 11, # calls
                                        'u': 12, # Unlabeled
                                        'B': 8, # bout_silence
                                        'S': 13 # silence
                                      },
                           },
                           
            'z_y19o20_21': {'len_motif': 0.45, 
                            'syllable_mapping': 
                                       {'1': 1, # s1
                                        '2': 2, # s2
                                        '3': 3, # s3
                                        '4': 4, # s4
                                        '5': 5, # s5
                                        '6': 6, # s6
                                        '7': 7, # s7
                                        'i': 10, # 
                                        'I': 10, # intro note
                                        'C': 11, # calls
                                        'u': 12, # Unlabeled
                                        'B': 8, # bout_silence
                                        'S': 13 # silence
                                      }, 
                             }
        }
    