Make sure you are into Finetuning SLMs directory: $cd "Finetuning SLMs"

Additional steps for T5_large
    Install redis, type the following commands in the terminal
        Step-1: sudo su
        Step-2: chmod +x redis_install.sh
        Step-3: ./redis_install.sh
        Step-4: redis-server redis.conf

For Finetuning SLM's
    Step - 1 Set up the environment

        First create conda environment
            $conda create --name slm python=3.9 -y
            $conda init bash
            $exec bash
            $conda activate slm

        For Decoder only models (phi-3, gpt2):
            Install requirements
                $pip install -r requirements.txt
            Finally go into the model folder which you want to train/inference
                e.g. $cd GPT2_medium
                For finetuning: $chmod +x finetune.sh
                                $./finetune.sh
                For generating inferencecs:
                                $chmod +x generate.shtab
                                $./generate.sh

        For Encoder-decoder models (T5):
            Install requirements
                $pip install -r requirements_T5.txt
            Go into the T5_large folder, $cd T5_large
            By default, data is expected to be in TSV format, with names as train_t5.tsv and val_t5.tsv, the path where these TSVs can be found is to passed as an argument of the upload command
            Upload data to redis
                $chmod +x automate_upload.sh
                $./automate_upload.sh 'path to data folder'
                eg -> ./automate_upload.sh ../data
                (depending on the data size it will take some time to upload)
            Finetune
                $chmod +x finetune.sh 
                $./finetune.sh 'number of GPU'
                eg -> ./finetune.sh 8
            Generate inferencecs from a single checkpoint
                $chmod +x generate.sh
                $./generate.sh 'checkpoint_path' 'test set path'
                eg -> ./generate.sh ./model_run_t5large/en_only/teacher/T5_ep50_bs4_ga2_gpus8/hf_checkpoints/epoch-0 ../data/test.tsv
            Sequentially generate inferencecs from all epochs (useful for hyperparameter search)
                $chmod +x generate_h.sh
                $./generate_h.sh 'checkpoint_folder_path' 'test set path'
                eg -> ./generate_h.sh .model_run_t5large/en_only/teacher/T5_ep50_bs4_ga2_gpus8/hf_checkpoints ../data/test.tsv
            Note - for t5 there might be an error due to the recent update in numpy library, to resolve you can simply change "np.Inf" to "np.inf" in the file that is mentioned in the error, however downgrading numpy would be preferred

                

