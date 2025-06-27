# fsl-3 main.py
import os
import torch
import sys
import traceback
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd


from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    BATCH_SIZE,
    LABEL_MAP,
    HOP_LENGTH,
    DEVICE, 
    N_SUPPORT, 
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_QUERY,
    TEST_SIZE,
    REQUIRED_CLASSES,
    N_WAY,
    EPISODES,
    PROTO_WEIGHT,
    RELATION_WEIGHT,
    EVALUATEAUDIO_DIR,
    EVALUATEDATAPATH,
    N_MELS,
    N_FFT
)

# Import preprocessing and training functions
from src.preprocess import (
    getmetadata,
    create_all_spectrograms,
    check_class_distribution,
    verify_few_shot_requirements,
    getexperimentdata,
    process_audio_file
)

from src.dataset import BirdSoundDataset, SegmentDataset
from src.models import CombinedFreqTemporalCNNEncoder, EnsembleModel
from src.training import train_few_shot
from src.evaluation import (
    evaluate_episodic, 
    update_metadata_results, 
    evaluate_ensemble_classification,
    update_segment_class_counts_with_time_aggregation,
    create_time_aggregated_summary
)

def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = getmetadata()
    
    # Create spectrograms for all training files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print("Class distribution:")
    for cls, count in dist["class_counts"].items():
        print(f" {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    return metadata_df

def preprocess_evaluation_data():
    """
    Prepare evaluation data by processing audio files into 1-second segments.
    """
    print("Preparing evaluation data...")
    
    # Get or create experiment metadata
    experiment_df = getexperimentdata()
    
    # Process any unprocessed files (this will create spectrograms for 1-second segments)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # For any unprocessed files, process them into segments
    unprocessed_files = experiment_df[experiment_df['processed'] == False]
    for idx, row in unprocessed_files.iterrows():
        try:
            file_path = row['file_path']
            _, segment_paths = process_audio_file(file_path, mel_spectrogram)
            
            if segment_paths:
                # Update paths in DataFrame
                experiment_df.at[idx, 'spectrogram_paths'] = ','.join(segment_paths)
                experiment_df.at[idx, 'processed'] = True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save updated DataFrame
    experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    return experiment_df

def main():
    # Step 1: Create directories if they don't exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    
    # Step 2: Run preprocessing
    metadata_df = preprocess_data()
    
    requirements = verify_few_shot_requirements(
        metadata_df, 
        n_way=N_WAY, 
        k_shot=N_SUPPORT,  
        query_size=N_QUERY, 
        test_size=TEST_SIZE
    )

    # Step 4: Prepare few-shot experiment
    if requirements["meets_requirements"]:
        all_metadata = metadata_df[metadata_df['label'].isin(REQUIRED_CLASSES)]
        all_dataset = BirdSoundDataset(all_metadata)

        # Create support dataset from training data for prototype creation
        train_metadata = all_metadata[~all_metadata['file_path'].str.contains('test/')]
        support_dataset = BirdSoundDataset(train_metadata)
        
        try:
            # Step 5: Initialize models
            encoder = CombinedFreqTemporalCNNEncoder().to(DEVICE)
            ensemble_model = EnsembleModel(encoder).to(DEVICE)
            
            # Step 6: Train the model
            print("Starting training...")
            train_losses = train_few_shot(
                model=ensemble_model,
                dataset=all_dataset,
                episodes=EPISODES,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                n_query=N_QUERY,
                relation_weight=RELATION_WEIGHT,
                proto_weight=PROTO_WEIGHT
            )
            
            # Step 7: Prepare evaluation data
            print("Preparing evaluation data...")
            experiment_df = preprocess_evaluation_data()
            
            # Create dataset of all 1-second segments for evaluation
            # We need to create a flat list of all spectrogram paths
            all_segment_paths = []
            for idx, row in experiment_df.iterrows():
                if row['processed'] and row['spectrogram_paths']:
                    segments = row['spectrogram_paths'].split(',')
                    all_segment_paths.extend(segments)
            
            if not all_segment_paths:
                print("No segments found for evaluation!")
                return
            
            # Create a DataFrame with just the paths for the segment dataset
            segments_df = pd.DataFrame({'file_path': all_segment_paths})
            evaluation_dataset = SegmentDataset(segments_df)
            
            # Step 8: Evaluate the model on segments
            print(f"Evaluating model on {len(evaluation_dataset)} segments...")
            results = evaluate_ensemble_classification(
                model=ensemble_model,
                segment_dataset=evaluation_dataset,
                support_dataset=support_dataset,
                device=DEVICE,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                batch_size=BATCH_SIZE
            )
            
            # Step 9: Update experiment DataFrame with time-aggregated segment class counts
            print("Updating experiment data with time-based aggregation...")
            experiment_df = update_segment_class_counts_with_time_aggregation(experiment_df, results)
            
            # Step 10: Create and save ONLY the time-aggregated summary
            print("Creating time-aggregated summary...")
            summary_df = create_time_aggregated_summary(experiment_df)
            
            # Save ONLY the summary to EVALUATEDATAPATH
            summary_df.to_csv(EVALUATEDATAPATH, index=False)
            print(f"Saved time-aggregated summary to {EVALUATEDATAPATH}")
            
            # Step 11: Update metadata with prediction results (if needed)
            # Only update if results contain files that are in the main metadata
            metadata_results = [r for r in results if not '_seg' in r.get('file_path', '')]
            if metadata_results:
                update_metadata_results(metadata_results, evaluation_dataset)
            
            print("Evaluation complete!")
            
            # Print summary statistics
            print(f"\nEvaluation Summary:")
            print(f"Total segments evaluated: {len(results)}")
            print(f"Total unique time periods: {len(summary_df)}")
            
            # Count predictions by class
            prediction_counts = {}
            for result in results:
                pred = result.get('prediction', 'unknown')
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            print(f"\nSegment-level predictions:")
            for class_name, count in prediction_counts.items():
                percentage = (count / len(results)) * 100
                print(f"  {class_name}: {count} segments ({percentage:.1f}%)")
            
            # Show time-aggregated statistics
            print(f"\nTime-aggregated statistics:")
            print(f"Average counts per time period:")
            for class_col in ['alarm_count_avg', 'non_alarm_count_avg', 'background_count_avg']:
                avg_count = summary_df[class_col].mean()
                print(f"  {class_col.replace('_count_avg', '')}: {avg_count:.1f}")
            
            # Show examples of time periods with multiple files
            multiple_files = summary_df[summary_df['num_files'] > 1]
            if len(multiple_files) > 0:
                print(f"\nTime periods with multiple files: {len(multiple_files)}")
                print("Examples:")
                for _, row in multiple_files.head(3).iterrows():
                    print(f"  {row['time_key']}: {row['num_files']} files, "
                          f"avg counts [{row['alarm_count_avg']}, {row['non_alarm_count_avg']}, {row['background_count_avg']}]")
                
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()  # Print the full stack trace for debugging
    else:
        print("Not enough data for few-shot learning.")
        print(requirements["suggestion"])

if __name__ == "__main__":
    main()