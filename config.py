mt5_path = "google/mt5-base"  # Use Hugging Face model directly

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-100.train",
                    "How2Sign": "./data/How2Sign/labels.train",
                    "OpenASL": "./data/OpenASL/labels.train",
                    "OSL-Words": "/home/sign_lang_fyp_sp26/fyp/FYPproject/final_split_osl_ssl/labels-mixed.train",
                    "OSL-Sentences": "./data/OSL-Sentences/labels-osl.train",
                    "OSL-Sentences-CV5": "./data/OSL-Sentences/labels-cv-fold5.train",
                    "OSL-Sentences-CV4": "./data/OSL-Sentences/labels-cv-fold4.train",
                    "OSL-Sentences-CV3": "./data/OSL-Sentences/labels-cv-fold3.train",
                    "OSL-Sentences-CV2": "./data/OSL-Sentences/labels-cv-fold2.train",
                    "OSL-Sentences-CV1": "./data/OSL-Sentences/labels-cv-fold1.train",
                    "OSL-Sentences-splitA": "./data/OSL-Sentences-splitA/labels-osl.train",
                    "OSL-Sentences-splitB": "./data/OSL-Sentences-splitB/labels-osl.train",
                    "OSL-Sentences-splitC": "./data/OSL-Sentences-splitC/labels-osl.train",
                    "OSL-Words-Webcam": "./data/OSL-Words-Webcam/labels-webcam.train",
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-100.dev",
                    "How2Sign": "",
                    "OpenASL": "./data/OpenASL/labels.dev",
                    "OSL-Words": "/home/sign_lang_fyp_sp26/fyp/FYPproject/final_split_osl_ssl/labels-mixed.dev",
                    "OSL-Sentences": "./data/OSL-Sentences/labels-osl.dev",
                    "OSL-Sentences-CV5": "./data/OSL-Sentences/labels-osl.dev",
                    "OSL-Sentences-CV4": "./data/OSL-Sentences/labels-osl.dev",
                    "OSL-Sentences-CV3": "./data/OSL-Sentences/labels-osl.dev",
                    "OSL-Sentences-CV2": "./data/OSL-Sentences/labels-osl.dev",
                    "OSL-Sentences-CV1": "./data/OSL-Sentences/labels-osl.dev",
                    "OSL-Sentences-splitA": "./data/OSL-Sentences-splitA/labels-osl.dev",
                    "OSL-Sentences-splitB": "./data/OSL-Sentences-splitB/labels-osl.dev",
                    "OSL-Sentences-splitC": "./data/OSL-Sentences-splitC/labels-osl.dev",
                    "OSL-Words-Webcam": "./data/OSL-Words-Webcam/labels-webcam.dev",
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-100.test",
                    "How2Sign": "./data/How2Sign/labels.test",
                    "OpenASL": "./data/OpenASL/labels.test",
                    "OSL-Words": "/home/sign_lang_fyp_sp26/fyp/FYPproject/final_split_osl_ssl/labels-mixed.test",
                    "OSL-Sentences": "./data/OSL-Sentences/labels-osl.test",
                    "OSL-Sentences-CV5": "./data/OSL-Sentences/labels-cv-fold5.test",
                    "OSL-Sentences-CV4": "./data/OSL-Sentences/labels-cv-fold4.test",
                    "OSL-Sentences-CV3": "./data/OSL-Sentences/labels-cv-fold3.test",
                    "OSL-Sentences-CV2": "./data/OSL-Sentences/labels-cv-fold2.test",
                    "OSL-Sentences-CV1": "./data/OSL-Sentences/labels-cv-fold1.test",
                    "OSL-Sentences-splitA": "./data/OSL-Sentences-splitA/labels-osl.test",
                    "OSL-Sentences-splitB": "./data/OSL-Sentences-splitB/labels-osl.test",
                    "OSL-Sentences-splitC": "./data/OSL-Sentences-splitC/labels-osl.test",
                    "OSL-Words-Webcam": "./data/OSL-Words-Webcam/labels-webcam.test",
}


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "How2Sign": "./dataset/How2Sign/rgb_format",
            "OpenASL": "./dataset/OpenASL/rgb_format",
            "OSL-Words": "/home/sign_lang_fyp_sp26/fyp/FYPproject/final_split_osl_ssl",
            "OSL-Sentences": "./dataset/OSL-Sentences",
            "OSL-Sentences-CV1": "./dataset/OSL-Sentences-CV1",
            "OSL-Sentences-splitA": "./dataset/OSL-Sentences-splitA",
            "OSL-Sentences-splitB": "./dataset/OSL-Sentences-splitB",
            "OSL-Sentences-splitC": "./dataset/OSL-Sentences-splitC",
            "OSL-Sentences-CV2": "./dataset/OSL-Sentences-CV2",
            "OSL-Sentences-CV3": "./dataset/OSL-Sentences-CV3",
            "OSL-Sentences-CV4": "./dataset/OSL-Sentences-CV4",
            "OSL-Sentences-CV5": "./dataset/OSL-Sentences-CV5",
            "OSL-Words-Webcam": "./dataset/OSL-Words/rgb_format",
            }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "How2Sign": "./dataset/WLASL/pose_format",
            "OpenASL": "./dataset/WLASL/pose_format",
            "OSL-Words": "/home/sign_lang_fyp_sp26/fyp/FYPproject/final_split_osl_ssl",
            "OSL-Sentences": "./dataset/OSL-Sentences",
            "OSL-Sentences-CV1": "./dataset/OSL-Sentences-CV1",
            "OSL-Sentences-splitA": "./dataset/OSL-Sentences-splitA",
            "OSL-Sentences-splitB": "./dataset/OSL-Sentences-splitB",
            "OSL-Sentences-splitC": "./dataset/OSL-Sentences-splitC",
            "OSL-Sentences-CV2": "./dataset/OSL-Sentences-CV2",
            "OSL-Sentences-CV3": "./dataset/OSL-Sentences-CV3",
            "OSL-Sentences-CV4": "./dataset/OSL-Sentences-CV4",
            "OSL-Sentences-CV5": "./dataset/OSL-Sentences-CV5",
            "OSL-Words-Webcam": "./dataset/OSL-Words-Webcam/pose_format",
}