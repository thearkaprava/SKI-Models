import os
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips

def load_sequences(sequence_file):
    """Load sequences from a text file."""
    with open(sequence_file, 'r', encoding='utf-8-sig') as file:
        sequences = [line.strip().split(',') for line in file.readlines()]
    return sequences

def find_videos(folder, action_id):
    """Find videos matching the action_id in the specified folder."""
    for file in os.listdir(folder):
        if file.startswith(f"A{int(action_id):03d}_") and file.endswith(".mp4"):
            return os.path.join(folder, file)
    return None

def process_sequences(base_path, output_base_path, sequences):
    """Process sequences and save concatenated videos in performer-specific folders."""
    for performer_id in os.listdir(base_path):
        performer_folder = os.path.join(base_path, performer_id)
        if not os.path.isdir(performer_folder):
            continue
        
        output_folder = os.path.join(output_base_path, performer_id)
        os.makedirs(output_folder, exist_ok=True)
        
        video_info = {}
        action_info = {}
        
        for idx, sequence in enumerate(sequences):
            clips = []
            for action_id in sequence:
                video_path = find_videos(performer_folder, action_id.strip())
                if video_path:
                    try:
                        clip = VideoFileClip(video_path)
                        if clip.duration > 0:
                            clips.append(clip)
                        else:
                            print(f"Clip {video_path} is invalid or has zero duration.")
                    except Exception as e:
                        print(f"Failed to load clip {video_path}: {e}")
                else:
                    print(f"Video for action ID {action_id} not found in {performer_folder}")
            
            if clips:
                try:
                    final_clip = concatenate_videoclips(clips)
                    output_filename = f"{performer_id}_video_{idx+1}.mp4"
                    final_clip_path = os.path.join(output_folder, output_filename)
                    final_clip.write_videofile(final_clip_path, codec='libx264')
                    final_clip.close()

                    video_info[output_filename] = [clip.filename for clip in clips]
                    action_info[output_filename] = [action_id.strip() for action_id in sequence]
                    print(f"Saved {output_filename} to {output_folder}")
                except Exception as e:
                    print(f"Error concatenating clips for sequence {idx+1}: {e}")
            else:
                print(f"No valid videos found for sequence {idx+1} in {performer_id}, skipping.")

        # Save JSON data for each performer
        with open(os.path.join(output_folder, 'video_data.json'), 'w') as f:
            json.dump(video_info, f, indent=4)
        with open(os.path.join(output_folder, 'action_data.json'), 'w') as f:
            json.dump(action_info, f, indent=4)

# Paths for the script to run
base_path = "path to your video folder"
output_base_path = "saving folder"
sequence_file = "action combination"

# Load sequences from the file and process them
sequences = load_sequences(sequence_file)
process_sequences(base_path, output_base_path, sequences)
