import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import StandardScaler
import os
import json
import hdbscan
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int16, Float32MultiArray # <--- Import new message type
import collections # Add this import at the top of your file

# Assuming lidar_processor.py is in the same directory
# This function must be identical to the one used in the training script.
from .lidar_processor import get_ranges_from_points

# --- Neural Network Models (Unchanged) ---
class LidarEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(LidarEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2d_3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(64, embedding_size)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class LidarDecoder(nn.Module):
    def __init__(self, embedding_size, num_ranges):
        super(LidarDecoder, self).__init__()
        self.num_ranges = num_ranges
        self.decoder_fc1 = nn.Linear(embedding_size, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, self.num_ranges)

    def forward(self, x):
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x) 
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ----------------------------------------------------------------------

class LiveClusterInferenceNode(Node):
    def __init__(self):
        super().__init__('live_cluster_inference_node')

        # --- Configuration: ROS parameters, overridden by the weights' own config.json
        #
        # The feature is `ranges / max_lidar_range` over a z band, so the encoder must
        # be fed exactly the preprocessing its checkpoint was fitted with.
        #   1. every value is a ROS parameter, so a weight set can be swapped without
        #      editing installed source;
        #   2. `config.json` written beside the weights by the trainer WINS over the
        #      parameters. A mismatch is logged.
        # Weight sets without a config.json (e.g. `bridge1_carla`) fall back to the
        # parameter defaults below, which are livox1 values and NOT what those weights were
        # fitted on: output is unreliable unless the correct preprocessing is passed.
        # The values in force are logged at startup.
        self.declare_parameter('training_data_name', 'livox1')
        self.declare_parameter('weights_dir', '')
        self.declare_parameter('embedding_size', 16)
        self.declare_parameter('num_ranges', 72)
        self.declare_parameter('max_lidar_range', 8.0)
        self.declare_parameter('min_lidar_range', 0.5)
        self.declare_parameter('z_threshold_lower', -0.5)
        self.declare_parameter('z_threshold_upper', 0.15)
        self.declare_parameter('z_threshold_lower_2', 0.0)
        self.declare_parameter('z_threshold_upper_2', 0.0)
        self.declare_parameter('use_intensity', False)
        self.declare_parameter('min_intensity', 32.0)
        self.declare_parameter('max_intensity', 68.0)
        self.declare_parameter('density_radius', 0.30)
        self.declare_parameter('min_neighbors', 4)
        self.declare_parameter('smoothing_window_size', 10)
        self.declare_parameter('distance_threshold_for_reassignment', 6.0)
        self.declare_parameter('cloud_topic', '/livox/lidar')

        def _p(name):
            return self.get_parameter(name).value

        training_data_name = _p('training_data_name')
        weights_dir = _p('weights_dir') or f"encoder_weights/{training_data_name}"

        cfg = {
            "num_ranges": int(_p('num_ranges')),
            "max_lidar_range": float(_p('max_lidar_range')),
            "min_lidar_range": float(_p('min_lidar_range')),
            "z_threshold_lower": float(_p('z_threshold_lower')),
            "z_threshold_upper": float(_p('z_threshold_upper')),
            "z_threshold_lower_2": float(_p('z_threshold_lower_2')),
            "z_threshold_upper_2": float(_p('z_threshold_upper_2')),
            "use_intensity": bool(_p('use_intensity')),
            "min_intensity": float(_p('min_intensity')),
            "max_intensity": float(_p('max_intensity')),
            "density_radius": float(_p('density_radius')),
            "min_neighbors": int(_p('min_neighbors')),
            "embedding_size": int(_p('embedding_size')),
        }

        cfg_path = os.path.join(weights_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                saved = json.load(f)
            differs = {k: (cfg.get(k), saved[k]) for k in saved
                       if k in cfg and cfg[k] != saved[k]}
            cfg.update({k: v for k, v in saved.items() if k in cfg})
            self.get_logger().info(f"loaded {cfg_path} (it wins over the parameters)")
            for k, (was, now) in differs.items():
                self.get_logger().warning(
                    f"  config.json overrides {k}: {was} -> {now}")
        else:
            self.get_logger().warning(
                f"no config.json in {weights_dir}. The preprocessing config is then "
                f"only as good as the parameters passed, and a wrong "
                f"max_lidar_range or z-band feeds the encoder an input it was never "
                f"fitted on -- silently, because every value is still a valid float. "
                f"Retrain with the current cluster_training.py to get one.")

        self.embedding_size = int(cfg.pop("embedding_size", 16))
        self.num_ranges = int(cfg["num_ranges"])
        self.max_lidar_range = float(cfg["max_lidar_range"])
        self.min_lidar_range = float(cfg["min_lidar_range"])
        self.angle_increment_deg = float(360.0 / self.num_ranges)
        self.z_threshold_upper = cfg["z_threshold_upper"]
        self.z_threshold_lower = cfg["z_threshold_lower"]
        self.z_threshold_upper_2 = cfg["z_threshold_upper_2"]
        self.z_threshold_lower_2 = cfg["z_threshold_lower_2"]
        cfg.setdefault("int_lower", 0.0)
        cfg.setdefault("int_upper", 255.0)
        cfg["angle_increment_deg"] = self.angle_increment_deg
        self.config = cfg

        self.smoothing_window_size = int(_p('smoothing_window_size'))
        self.prediction_buffer = []
        self.override_label = None
        self.distance_threshold_for_reassignment = float(
            _p('distance_threshold_for_reassignment'))

        # THE VALUE IN FORCE, at startup, every run.
        self.get_logger().info(
            f"cluster config: weights={weights_dir} num_ranges={self.num_ranges} "
            f"max_range={self.max_lidar_range} "
            f"z=[{self.z_threshold_lower},{self.z_threshold_upper}] "
            f"intensity={self.config['use_intensity']} "
            f"embedding={self.embedding_size} smoothing={self.smoothing_window_size}")

        path_prefix = ""

        self.model_save_path = os.path.join(
            weights_dir, f"lidar_encoder_autoencoder_{training_data_name}.pth")
        self.hdbscan_model_path = os.path.join(
            weights_dir, f"hdbscan_model_{training_data_name}.pkl")
        self.scaler_path = os.path.join(
            weights_dir, f"scaler_{training_data_name}.pkl")
        self.cluster_centroids_path = os.path.join(
            weights_dir, f"cluster_centroids_{training_data_name}.pkl")

        # --- Device Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- Load Pre-Trained Models and Centroids ---
        self.load_models()

        # --- Subscription and Publishing ---
        self.subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter('cloud_topic').value,
            self.pointcloud_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.publisher_ = self.create_publisher(Int16, '/predicted_cluster', 10)
        self.publisher_ranges = self.create_publisher(Float32MultiArray, '/processed_ranges', 10)
        self.override_sub = self.create_subscription(Int16, '/cluster_override', self._override_cb, 10)
        self.filter_cfg_sub = self.create_subscription(
            Float32MultiArray, '/lidar_filter_config', self._cb_filter_cfg, 10)

        self.get_logger().info(
            f"Ready for live clustering. Subscribed to "
            f"'{self.subscription.topic_name}' and publishing to '{self.publisher_.topic_name}'")

    def load_models(self):
        """Loads the pre-trained Autoencoder, HDBSCAN model, StandardScaler, and Cluster Centroids."""
        encoder = LidarEncoder(embedding_size=self.embedding_size)
        decoder = LidarDecoder(embedding_size=self.embedding_size, num_ranges=self.num_ranges)
        self.autoencoder = Autoencoder(encoder, decoder).to(self.device)

        if os.path.exists(self.model_save_path):
            # strict=False: the DECODER is never used at inference, so a checkpoint
            # whose decoder width does not match (bridge1_carla is 32 where the default
            # is 72) should still load its encoder rather than refuse to start.
            missing, unexpected = self.autoencoder.load_state_dict(
                torch.load(self.model_save_path, map_location=self.device),
                strict=False)
            enc_bad = [k for k in list(missing) + list(unexpected)
                       if k.startswith('encoder.')]
            if enc_bad:
                self.get_logger().error(
                    f"ENCODER weights do not match the model: {enc_bad}. The embedding "
                    f"would be nonsense; refusing to run.")
                exit()
            if missing or unexpected:
                self.get_logger().warning(
                    f"decoder keys ignored (unused at inference): "
                    f"{list(missing) + list(unexpected)}")
            self.autoencoder.eval()
            self.get_logger().info(f"Successfully loaded Autoencoder model from {self.model_save_path}")
        else:
            self.get_logger().error(f"Autoencoder model not found at {self.model_save_path}! Exiting.")
            exit()
        
        if os.path.exists(self.hdbscan_model_path):
            with open(self.hdbscan_model_path, 'rb') as f:
                self.cluster_model = pickle.load(f)
            self.get_logger().info(f"Successfully loaded HDBSCAN model from {self.hdbscan_model_path}")
        else:
            self.get_logger().error(f"HDBSCAN model not found at {self.hdbscan_model_path}! Exiting.")
            exit()
            
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.get_logger().info(f"Successfully loaded StandardScaler from {self.scaler_path}")
        else:
            self.get_logger().error(f"StandardScaler not found at {self.scaler_path}! Exiting.")
            exit()

        if os.path.exists(self.cluster_centroids_path):
            with open(self.cluster_centroids_path, 'rb') as f:
                self.cluster_centroids = pickle.load(f)
            self.get_logger().info(f"Successfully loaded cluster centroids from {self.cluster_centroids_path}")
        else:
            self.get_logger().error(f"Cluster centroids not found at {self.cluster_centroids_path}! Please re-run the training script with logic to save them. Exiting.")
            exit()
            
    def _get_smoothed_label(self, new_label):
        """
        Applies a mode filter to the predicted cluster labels to smooth out
        rapid changes.
        """
        # Add the new label to the buffer
        self.prediction_buffer.append(new_label)

        # Keep the buffer size within the defined window
        if len(self.prediction_buffer) > self.smoothing_window_size:
            self.prediction_buffer.pop(0)

        # Return the mode (most common value) of the buffer
        if not self.prediction_buffer:
            return -1 # Return a default value if buffer is empty

        # Use collections.Counter to handle negative integers
        counts = collections.Counter(self.prediction_buffer)
        # The most_common(1) method returns a list of (element, count) tuples.
        # We want the first element's value.
        most_common_label = counts.most_common(1)[0][0]
        
        return int(most_common_label)
    
    def reassign_labels(self, hdbscan_label, embedding):
        """
        Reassigns a label if HDBSCAN classified it as noise.
        """
        if hdbscan_label != -1:
            return hdbscan_label # No change for non-noise points

        if not self.cluster_centroids:
            return -1 # Cannot reassign if no centroids are loaded

        distances_to_all_centroids = {
            cid: np.linalg.norm(embedding - centroid)
            for cid, centroid in self.cluster_centroids.items()
        }

        closest_known_cluster_id = min(distances_to_all_centroids, key=distances_to_all_centroids.get)
        min_dist = distances_to_all_centroids[closest_known_cluster_id]

        if min_dist < self.distance_threshold_for_reassignment:
            # self.get_logger().info(f"Reassigning noise point to cluster {closest_known_cluster_id} with distance {min_dist:.2f}")
            return closest_known_cluster_id
        else:
            # self.get_logger().info(f"Classifying point as a new cluster/outlier (ID -2) with distance {min_dist:.2f}")
            return -2

    def _cb_filter_cfg(self, msg: Float32MultiArray):
        # Message layout: [z_upper, z_lower, z2_upper, z2_lower, max_range, min_range, use_intensity, int_lower, int_upper]
        if len(msg.data) < 6:
            return
        self.config['z_threshold_upper']   = float(msg.data[0])
        self.config['z_threshold_lower']   = float(msg.data[1])
        self.config['z_threshold_upper_2'] = float(msg.data[2])
        self.config['z_threshold_lower_2'] = float(msg.data[3])
        self.config['max_lidar_range']     = float(msg.data[4])
        self.config['min_lidar_range']     = float(msg.data[5])
        self.max_lidar_range               = float(msg.data[4])  # kept in sync for normalization
        if len(msg.data) >= 7:
            self.config['use_intensity'] = bool(msg.data[6])
        if len(msg.data) >= 9:
            self.config['int_lower'] = float(msg.data[7])
            self.config['int_upper'] = float(msg.data[8])
        if len(msg.data) >= 11:
            self.config['density_radius'] = float(msg.data[9])
            self.config['min_neighbors']  = int(msg.data[10])
        self.get_logger().info(
            f"Filter updated: mode={'intensity' if self.config['use_intensity'] else 'z-height'}  "
            f"z=[{msg.data[1]:.2f},{msg.data[0]:.2f}]  "
            f"r=[{msg.data[5]:.2f},{msg.data[4]:.2f}]")

    def _override_cb(self, msg: Int16):
        if msg.data == -99:
            self.override_label = None
            self.get_logger().info("Cluster override cleared.")
        else:
            self.override_label = msg.data
            self.get_logger().info(f"Cluster override set to {msg.data}.")

    def pointcloud_callback(self, msg: PointCloud2):
        """Callback for new PointCloud2 messages. Processes the data and performs inference."""
        # --- REMOVED: Performance-intensive logging ---
        # self.get_logger().info(f"Received PointCloud2 with {msg.width * msg.height} points.")

        try:
            has_intensity = any(f.name == 'intensity' for f in msg.fields)
            field_names = ("x", "y", "z", "intensity") if has_intensity else ("x", "y", "z")
            points_gen = read_points(msg, field_names=field_names)
            dtype = ([('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
                     if has_intensity else
                     [('x', np.float32), ('y', np.float32), ('z', np.float32)])
            points_structured = np.asarray(list(points_gen), dtype=dtype)
        except Exception as e:
            self.get_logger().error(f"Error reading points from PointCloud2: {e}")
            return

        if points_structured.shape[0] == 0:
            # self.get_logger().warn("Received empty PointCloud2 message. Publishing -1 (Noise).")
            label_msg = Int16()
            label_msg.data = -1
            self.publisher_.publish(label_msg)
            return

        intensity_col = (points_structured['intensity'].astype(np.float32)
                         if has_intensity else np.zeros(len(points_structured), np.float32))
        points_np = np.column_stack([
            points_structured['x'], points_structured['y'],
            points_structured['z'], intensity_col,
        ])
        
        # Use the unified pre-processing function to generate the 1D range array
        ranges_binned = get_ranges_from_points(points_np, self.config)
        
        if np.all(ranges_binned == self.config['max_lidar_range']):
            # self.get_logger().warn("All ranges are at max_lidar_range. Publishing -1 (Noise).")
            label_msg = Int16()
            label_msg.data = -1
            self.publisher_.publish(label_msg)
            return
        
        ranges_msg = Float32MultiArray()
        ranges_msg.data = ranges_binned.tolist()
        self.publisher_ranges.publish(ranges_msg)

        normalized_ranges = ranges_binned / self.max_lidar_range
        input_ranges = torch.tensor(normalized_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.autoencoder.encoder(input_ranges).cpu().numpy()
        
        scaled_embedding = self.scaler.transform(embedding)
        
        predicted_label, _ = hdbscan.approximate_predict(self.cluster_model, scaled_embedding)
        
        final_label = self.reassign_labels(int(predicted_label[0]), scaled_embedding[0])
        smoothed_label = self._get_smoothed_label(int(final_label))

        label_msg = Int16()
        label_msg.data = self.override_label if self.override_label is not None else smoothed_label
        self.publisher_.publish(label_msg)

        # --- REMOVED: Performance-intensive logging ---
        # self.get_logger().info(f"Published Predicted Cluster: {label_msg.data}")
        
def main(args=None):
    rclpy.init(args=args)
    inference_node = LiveClusterInferenceNode()
    try:
        rclpy.spin(inference_node)
    except KeyboardInterrupt:
        pass
    inference_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from sensor_msgs_py.point_cloud2 import read_points
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pickle
# from sklearn.preprocessing import StandardScaler
# import os
# import hdbscan
# from rclpy.qos import qos_profile_sensor_data
# from std_msgs.msg import Int16, Float32MultiArray
# import umap
# import joblib

# # Assuming lidar_processor.py is in the same directory
# from .lidar_processor import get_ranges_from_points

# # --- Neural Network Models (Unchanged) ---
# class LidarEncoder(nn.Module):
#     def __init__(self, embedding_size):
#         super(LidarEncoder, self).__init__()
#         self.embedding_size = embedding_size
#         self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
#         self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         self.conv2d_3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         self.pool = nn.AdaptiveMaxPool2d((1,1))
#         self.fc = nn.Linear(64, embedding_size)

#     def forward(self, x):
#         x = x.unsqueeze(1).unsqueeze(1)
#         x = F.relu(self.conv2d_1(x))
#         x = F.relu(self.conv2d_2(x))
#         x = F.relu(self.conv2d_3(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         embedding = self.fc(x)
#         return embedding

# class LidarDecoder(nn.Module):
#     def __init__(self, embedding_size, num_ranges):
#         super(LidarDecoder, self).__init__()
#         self.num_ranges = num_ranges
#         self.decoder_fc1 = nn.Linear(embedding_size, 64)
#         self.decoder_fc2 = nn.Linear(64, 128)
#         self.decoder_fc3 = nn.Linear(128, self.num_ranges)

#     def forward(self, x):
#         x = F.relu(self.decoder_fc1(x))
#         x = F.relu(self.decoder_fc2(x))
#         x = self.decoder_fc3(x) 
#         return x

# class Autoencoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Autoencoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # ----------------------------------------------------------------------

# class LiveClusterInferenceNode(Node):
#     def __init__(self):
#         super().__init__('live_cluster_inference_node')

#         # --- Configuration Parameters (must match training script) ---
#         training_data_name = "full_bag"
#         self.embedding_size = 16 # This is the size for a single autoencoder
#         self.num_ranges = 32
#         self.max_lidar_range1 = 8 # Set this to match autoencoder 1 training
#         self.max_lidar_range2 = 25 # Set this to match autoencoder 2 training
#         self.angle_increment_deg = float(360.0 / self.num_ranges)
        
#         # Multiple altitude slices to match training data preprocessing
#         self.z_threshold_upper = 2.25
#         self.z_threshold_lower = 2
#         self.z_threshold_upper_2 = 0
#         self.z_threshold_lower_2 = 0

#         self.distance_threshold_for_reassignment = 6

#         self.config = {
#             "num_ranges": self.num_ranges,
#             "max_lidar_range1": self.max_lidar_range1,
#             "max_lidar_range2": self.max_lidar_range2,
#             "angle_increment_deg": self.angle_increment_deg,
#             "z_threshold_upper": self.z_threshold_upper,
#             "z_threshold_lower": self.z_threshold_lower,
#             "z_threshold_upper_2": self.z_threshold_upper_2,
#             "z_threshold_lower_2": self.z_threshold_lower_2
#         }
        
#         self.final_label_mapping = {
#             "Road": 10,
#             "Past Building": 9,
#             "Around Corner": 8,
#             "Exit Intersection/Enter Bridge": 1,
#             "Along Wall": 7,
#             "In Intersection": 6,
#             "Enter Intersection/Exit Bridge": 4,
#             "Open Space": 0,
#             "On Bridge": 3,
#             "unlabeled": -1, # or any other desired value for unlabeled data
#         }
        
#         self.cluster_to_description_map = {
#             -1: "Road",
#             "0": "Past Building",
#             "1": "Around Corner",
#             "2": "Around Corner",
#             "3": "Exit Intersection/Enter Bridge",
#             "4": "Along Wall",
#             "5": "Along Wall",
#             "6": "Along Wall",
#             "7": "Along Wall",
#             "8": "Along Wall",
#             "9": "Along Wall",
#             "10": "Along Wall",
#             "11": "In Intersection",
#             "12": "Enter Intersection/Exit Bridge",
#             "13": "Enter Intersection/Exit Bridge",
#             "14": "Enter Intersection/Exit Bridge",
#             "15": "In Intersection",
#             "16": "Open Space",
#             "17": "In Intersection",
#             "18": "Exit Intersection/Enter Bridge",
#             "19": "Enter Intersection/Exit Bridge",
#             "20": "Enter Intersection/Exit Bridge",
#             "21": "Open Space",
#             "22": "Open Space",
#             "23": "Open Space",
#             "24": "Enter Intersection/Exit Bridge",
#             "25": "In Intersection",
#             "26": "Road",
#             "27": "In Intersection",
#             "28": "In Intersection",
#             "29": "On Bridge",
#             "30": "Road",
#             "31": "On Bridge",
#             "32": "Road",
#             "33": "Road",
#             "34": "Road",
#             "35": "Exit Intersection/Enter Bridge",
#             "36": "unlabeled",
#             "37": "Road",
#             "38": "Road",
#             "39": "unlabeled",
#             "40": "unlabeled",
#             "41": "unlabeled",
#             "42": "On Bridge",
#             "43": "On Bridge",
#             "44": "Road",
#             "45": "unlabeled",
#             "46": "unlabeled",
#             "47": "On Bridge",
#             "48": "On Bridge",
#             "49": "Road",
#             "50": "Road",
#             "51": "Road"
#         }
        
#         path_prefix = ""

#         # --- NEW: Define paths for all models in the new pipeline ---
#         self.autoencoder1_path = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}_{self.max_lidar_range1}.pth"
#         self.autoencoder2_path = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}_{self.max_lidar_range2}.pth"
#         self.umap_reducer_path = f"encoder_weights/{training_data_name}/umap_{training_data_name}.pkl"
#         self.hdbscan_model_path = f"encoder_weights/{training_data_name}/hdbscan_model_{training_data_name}.pkl"
#         self.scaler_path = f"encoder_weights/{training_data_name}/scaler_{training_data_name}.pkl"
#         self.cluster_centroids_path = f"encoder_weights/{training_data_name}/cluster_centroids_{training_data_name}.pkl"

#         # --- Device Setup ---
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.get_logger().info(f"Using device: {self.device}")

#         # --- Load All Models and Centroids ---
#         self.load_models()

#         # --- Subscription and Publishing ---
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             '/carla/ego_vehicle/lidar',
#             self.pointcloud_callback,
#             qos_profile=qos_profile_sensor_data
#         )
#         self.publisher_ = self.create_publisher(Int16, '/predicted_cluster', 10)
#         self.publisher_ranges = self.create_publisher(Float32MultiArray, '/processed_ranges', 10)

#         self.get_logger().info("Ready for live clustering. Subscribed to '/carla/ego_vehicle/lidar' and publishing to '/predicted_cluster'")

#     def load_models(self):
#         """Loads all models from the new pipeline."""
        
#         # --- Load Autoencoder 1 ---
#         encoder1 = LidarEncoder(embedding_size=self.embedding_size)
#         decoder1 = LidarDecoder(embedding_size=self.embedding_size, num_ranges=self.num_ranges)
#         self.autoencoder1 = Autoencoder(encoder1, decoder1).to(self.device)
#         if os.path.exists(self.autoencoder1_path):
#             self.autoencoder1.load_state_dict(torch.load(self.autoencoder1_path, map_location=self.device))
#             self.autoencoder1.eval()
#             self.get_logger().info(f"Successfully loaded Autoencoder model 1 from {self.autoencoder1_path}")
#         else:
#             self.get_logger().error(f"Autoencoder 1 not found at {self.autoencoder1_path}! Exiting.")
#             exit()
        
#         # --- Load Autoencoder 2 ---
#         encoder2 = LidarEncoder(embedding_size=self.embedding_size)
#         decoder2 = LidarDecoder(embedding_size=self.embedding_size, num_ranges=self.num_ranges)
#         self.autoencoder2 = Autoencoder(encoder2, decoder2).to(self.device)
#         if os.path.exists(self.autoencoder2_path):
#             self.autoencoder2.load_state_dict(torch.load(self.autoencoder2_path, map_location=self.device))
#             self.autoencoder2.eval()
#             self.get_logger().info(f"Successfully loaded Autoencoder model 2 from {self.autoencoder2_path}")
#         else:
#             self.get_logger().error(f"Autoencoder 2 not found at {self.autoencoder2_path}! Exiting.")
#             exit()

#         # --- Load HDBSCAN Model ---
#         if os.path.exists(self.hdbscan_model_path):
#             with open(self.hdbscan_model_path, 'rb') as f:
#                 self.cluster_model = pickle.load(f)
#             self.get_logger().info(f"Successfully loaded HDBSCAN model from {self.hdbscan_model_path}")
#         else:
#             self.get_logger().error(f"HDBSCAN model not found at {self.hdbscan_model_path}! Exiting.")
#             exit()

            
#         # --- Load StandardScaler ---
#         if os.path.exists(self.scaler_path):
#             with open(self.scaler_path, 'rb') as f:
#                 self.scaler = pickle.load(f)
#             self.get_logger().info(f"Successfully loaded StandardScaler from {self.scaler_path}")
#         else:
#             self.get_logger().error(f"StandardScaler not found at {self.scaler_path}! Exiting.")
#             exit()
    
#         # --- Load Cluster Centroids ---
#         if os.path.exists(self.cluster_centroids_path):
#             with open(self.cluster_centroids_path, 'rb') as f:
#                 self.cluster_centroids = pickle.load(f)
#             self.get_logger().info(f"Successfully loaded cluster centroids from {self.cluster_centroids_path}")
#         else:
#             self.get_logger().error(f"Cluster centroids not found at {self.cluster_centroids_path}! Please re-run the training script with logic to save them. Exiting.")
#             exit()
            
#         # # --- Load UMAP Reducer ---
#         # self.get_logger().info("before umap")
#         # if os.path.exists(self.umap_reducer_path):
#         #     self.get_logger().info("inside umap")
#         #     with open(self.umap_reducer_path, 'rb') as f:
#         #         self.get_logger().info(f"before pickle {self.umap_reducer_path}")
#         #         self.umap_reducer = pickle.load(f)
#         #         self.get_logger().info("after pickle")
#         #     self.get_logger().info(f"Successfully loaded UMAP reducer from {self.umap_reducer_path}")
#         # else:
#         #     self.get_logger().error(f"UMAP reducer not found at {self.umap_reducer_path}! Exiting.")
#         #     exit()
        
#         # random_state = 42
#         # self.reducer = umap.UMAP(n_neighbors=15, n_components=5, random_state=random_state)
            
#     def reassign_labels(self, hdbscan_label, embedding):
#         """
#         Reassigns a label if HDBSCAN classified it as noise.
#         """
#         if hdbscan_label != -1:
#             return hdbscan_label

#         if not self.cluster_centroids:
#             return -1

#         distances_to_all_centroids = {
#             cid: np.linalg.norm(embedding - centroid)
#             for cid, centroid in self.cluster_centroids.items()
#         }

#         closest_known_cluster_id = min(distances_to_all_centroids, key=distances_to_all_centroids.get)
#         min_dist = distances_to_all_centroids[closest_known_cluster_id]

#         if min_dist < self.distance_threshold_for_reassignment:
#             return closest_known_cluster_id
#         else:
#             return -2
        
#     def pointcloud_callback(self, msg: PointCloud2):
#         """Callback for new PointCloud2 messages. Processes the data and performs inference."""
        
#         try:
#             points_gen = read_points(msg, field_names=("x", "y", "z"))
#             points_structured = np.asarray(list(points_gen), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
#         except Exception as e:
#             self.get_logger().error(f"Error reading points from PointCloud2: {e}")
#             return
        
#         if points_structured.shape[0] == 0:
#             label_msg = Int16()
#             label_msg.data = -1
#             self.publisher_.publish(label_msg)
#             return

#         points_np = np.vstack([points_structured['x'], points_structured['y'], points_structured['z']]).T
        
#         # --- Pre-processing and embedding generation for concatenated models ---
#         ranges_binned1 = get_ranges_from_points(points_np, self.config, max_range=self.config['max_lidar_range1'])
#         normalized_ranges1 = ranges_binned1 / self.config['max_lidar_range1']
#         input_ranges1 = torch.tensor(normalized_ranges1, dtype=torch.float32).unsqueeze(0).to(self.device)

#         ranges_binned2 = get_ranges_from_points(points_np, self.config, max_range=self.config['max_lidar_range2'])
#         normalized_ranges2 = ranges_binned2 / self.config['max_lidar_range2']
#         input_ranges2 = torch.tensor(normalized_ranges2, dtype=torch.float32).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             embedding1 = self.autoencoder1.encoder(input_ranges1).cpu().numpy()
#             embedding2 = self.autoencoder2.encoder(input_ranges2).cpu().numpy()

#         concatenated_embedding = np.concatenate((embedding1, embedding2), axis=1)
        
#         loaded_reducer = joblib.load('encoder_weights/full_bag/umap_full_bag.joblib')
#         umap_embedding = loaded_reducer.transform(concatenated_embedding)
    
#         #umap_embedding = self.umap_reducer.transform(concatenated_embedding)
        
#         scaled_embedding = self.scaler.transform(umap_embedding)
        
#         predicted_label, _ = hdbscan.approximate_predict(self.cluster_model, scaled_embedding)
        
#         final_label = self.reassign_labels(int(predicted_label[0]), scaled_embedding[0])
        
#         descriptive_label = self.cluster_to_description_map.get(str(final_label), "unlabeled")

#         # Get the final integer ID from the new mapping
#         final_numeric_label = self.final_label_mapping.get(descriptive_label, -1)

#         label_msg = Int16()
#         #label_msg.data = int(final_label)
#         label_msg.data = int(final_numeric_label)
#         self.publisher_.publish(label_msg)
        
# def main(args=None):
#     rclpy.init(args=args)
#     try:
#         inference_node = LiveClusterInferenceNode()
#         rclpy.spin(inference_node)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
