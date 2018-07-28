
# coding: utf-8

# # Plot prediction images
# 
# For turning

# In[13]:


from utils import plotting_tools
import numpy as np
import os.path

def show_samples(grading_data_dir_name, subset_name, num_of_samples = 100, run_num = 'run_1'):
    """Predicted Samples Viewer"""
    # Count all iamges
    path = os.path.join('..', 'data', grading_data_dir_name)
    ims = np.array(plotting_tools.get_im_files(path, subset_name))
    print('All images: ', len(ims))

    # Show samples
    im_files = plotting_tools.get_im_file_sample(
        grading_data_dir_name, subset_name, run_num, n_file_names=num_of_files)
    print('Sample images: ', len(im_files))

    for i in range(min(num_of_files, len(ims))):
        im_tuple = plotting_tools.load_images(im_files[i])
        plotting_tools.show_images(im_tuple)
        
show_samples(grading_data_dir_name='sample_evaluation_data', subset_name='patrol_non_targ')

