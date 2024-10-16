import os
import streamlit as st
from constants import folders

def cache_management():
    if st.button("Refresh Cache View"):
        st.experimental_rerun()

    for HTOL_name in folders:
        with st.expander(f"Cached Files for {HTOL_name}"):
            if st.button(f"Delete All Cache for {HTOL_name}", key=f"del_file_machine_{HTOL_name}"):
                for hyperparameter_hash in hyperparameter_combinations:
                    cache_subdir = os.path.join('.cache', HTOL_name, hyperparameter_hash)
                    for cached_file in os.listdir(cache_subdir):
                        os.remove(os.path.join(cache_subdir, cached_file))
                st.experimental_rerun()

            hyperparameter_combinations = set()
            for root, dirs, files in os.walk(os.path.join('.cache', HTOL_name)):
                for dir_name in dirs:
                    hyperparameter_combinations.add(dir_name)

            for hyperparameter_hash in hyperparameter_combinations:
                cache_subdir = os.path.join('.cache', HTOL_name, hyperparameter_hash)
                cached_files = [f for f in os.listdir(cache_subdir) if f.endswith('.pkl')]

                if cached_files:
                    st.write(f"**Hyperparameters:** {hyperparameter_hash}")

                    if st.button(f"Delete All Cache for these Hyperparameters", key=f"del_file_hyper_{HTOL_name}_{hyperparameter_hash}"):
                        for cached_file in cached_files:
                            os.remove(os.path.join(cache_subdir, cached_file))
                        st.experimental_rerun()

                    for cached_file in cached_files:
                        file_name = cached_file.split('_')[0]  # Extract original file name
                        if st.button(f"Delete Cache for {file_name}", key=f"del_file_specific_{HTOL_name}_{file_name}_{hyperparameter_hash}"):
                            os.remove(os.path.join(cache_subdir, cached_file))
                            st.experimental_rerun()