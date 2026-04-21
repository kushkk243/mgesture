#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

class GestureType(Enum):
    NONE = "none"
    BRUSHING = "brushing"
    PATTING = "patting"
    POKING = "poking"
    HUGGING = "hugging"
    RUBBING = "rubbing"  # NEW


@dataclass
class GestureConfig:
    """Configuration tuned for 15 FPS input"""
    
    # Timing
    window_duration: float = 2.0
    sample_rate: float = 15.0
    
    # Brushing thresholds
    brush_direction_consistency: float = 0.7
    brush_min_strokes: int = 2
    
    # Patting thresholds
    pat_min_events: int = 2
    pat_area_variance_max: float = 2.0
    pat_min_coverage: float = 0.25
    
    # Poking thresholds
    poke_max_cells: int = 1
    poke_min_events: int = 2
    poke_location_consistency: float = 0.75
    
    # Hugging thresholds
    hug_min_coverage: float = 0.5
    hug_stability: float = 0.8
    
    # Rubbing thresholds (NEW)
    rub_min_direction_changes: int = 2       # At least 2 back-and-forth cycles
    rub_min_movements: int = 6               # Minimum movement detections
    rub_continuity: float = 0.6              # Should be mostly continuous contact
    rub_bidirectional_balance: float = 0.25  # Each direction at least 25% of movements
    
    # Binarization
    binary_threshold: int = 1


class GestureAnalyzer:
    """Analyzes buffered frames to detect gestures"""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        
    def analyze(self, frames: List[np.ndarray], timestamps: List[float]) -> Tuple[GestureType, float, dict]:
        """
        Analyze frame buffer and return detected gesture with confidence
        """
        if len(frames) < 5:
            return GestureType.NONE, 0.0, {}
        
        features = self._extract_features(frames, timestamps)
        
        # Check each gesture (order matters for disambiguation)
        # Rubbing before brushing since rubbing is more specific
        checks = [
            (self._check_hugging(features), GestureType.HUGGING),
            (self._check_poking(features), GestureType.POKING),
            (self._check_rubbing(features), GestureType.RUBBING),   # Check before brushing
            (self._check_brushing(features), GestureType.BRUSHING),
            (self._check_patting(features), GestureType.PATTING),
        ]
        
        best_confidence, best_gesture = max(checks, key=lambda x: x[0])
        
        debug_info = {
            'all_scores': {g.value: round(c, 3) for c, g in checks},
            'n_frames': features['n_frames'],
            'active_ratio': round(features['active_ratio'], 3),
            'mean_coverage': round(features['mean_coverage'], 3),
            'n_activations': features['n_activations'],
            'direction_changes': features['direction_changes'],
            'directions': features['directions'][-10:],  # Last 10 for debug
        }
        
        if best_confidence > 0.5:
            return best_gesture, best_confidence, debug_info
        return GestureType.NONE, 0.0, debug_info
    
    def _extract_features(self, frames: List[np.ndarray], timestamps: List[float]) -> dict:
        """Extract all relevant features from frame buffer"""
        frames_array = np.array(frames)
        n_frames = len(frames)
        total_cells = 8.0
        
        # Coverage features
        coverage_per_frame = np.sum(frames_array, axis=(1, 2)) / total_cells
        active_cells_per_frame = np.sum(frames_array, axis=(1, 2))
        
        # Activation events
        is_active = coverage_per_frame > 0
        activation_starts = np.where(np.diff(is_active.astype(int)) == 1)[0] + 1
        
        if is_active[0]:
            activation_starts = np.concatenate([[0], activation_starts])
        
        # Centroid tracking
        centroids = []
        for frame in frames_array:
            if np.sum(frame) > 0:
                rows, cols = np.where(frame == 1)
                centroids.append((np.mean(rows), np.mean(cols)))
            else:
                centroids.append(None)
        
        # Direction computation with change detection
        directions, direction_changes = self._compute_directions_with_changes(centroids)
        
        # Cell activation frequency
        cell_activation_count = np.sum(frames_array, axis=0)
        
        # State changes
        state_changes = np.sum(np.abs(np.diff(is_active.astype(int))))
        
        # Coverage variance
        active_frame_cells = active_cells_per_frame[active_cells_per_frame > 0]
        coverage_variance = np.var(active_frame_cells) if len(active_frame_cells) > 1 else 0
        
        return {
            'frames': frames_array,
            'n_frames': n_frames,
            'coverage_per_frame': coverage_per_frame,
            'active_cells_per_frame': active_cells_per_frame,
            'mean_coverage': np.mean(coverage_per_frame),
            'coverage_variance': coverage_variance,
            'is_active': is_active,
            'activation_starts': activation_starts,
            'n_activations': len(activation_starts),
            'centroids': centroids,
            'directions': directions,
            'direction_changes': direction_changes,
            'cell_activation_count': cell_activation_count,
            'state_changes': state_changes,
            'active_ratio': np.mean(is_active),
        }
    
    def _compute_directions_with_changes(
        self, 
        centroids: List[Optional[Tuple[float, float]]]
    ) -> Tuple[List[str], int]:
        """
        Compute movement directions and count direction reversals
        Returns: (directions_list, number_of_direction_changes)
        """
        directions = []
        prev_centroid = None
        
        for centroid in centroids:
            if centroid is None:
                prev_centroid = None
                continue
            
            if prev_centroid is not None:
                dr = centroid[0] - prev_centroid[0]
                dc = centroid[1] - prev_centroid[1]
                
                # Determine primary direction (horizontal or vertical)
                if abs(dc) > abs(dr) and abs(dc) > 0.2:
                    directions.append('right' if dc > 0 else 'left')
                elif abs(dr) > 0.2:
                    directions.append('down' if dr > 0 else 'up')
                # else: no significant movement
            
            prev_centroid = centroid
        
        # Count direction changes (reversals)
        direction_changes = 0
        if len(directions) >= 2:
            for i in range(1, len(directions)):
                curr = directions[i]
                prev = directions[i-1]
                
                # Check for reversal (opposite directions)
                is_reversal = (
                    (curr == 'left' and prev == 'right') or
                    (curr == 'right' and prev == 'left') or
                    (curr == 'up' and prev == 'down') or
                    (curr == 'down' and prev == 'up')
                )
                
                if is_reversal:
                    direction_changes += 1
        
        return directions, direction_changes
    
    def _check_rubbing(self, features: dict) -> float:
        """
        Rubbing: Back-and-forth movement pattern
        
        Key characteristics:
        - Bidirectional movement (oscillating)
        - Multiple direction reversals
        - Relatively continuous contact
        - Both directions should be represented
        """
        directions = features['directions']
        direction_changes = features['direction_changes']
        active_ratio = features['active_ratio']
        
        # Need enough movement data
        if len(directions) < self.config.rub_min_movements:
            return 0.0
        
        # Must have direction reversals (back-and-forth)
        if direction_changes < self.config.rub_min_direction_changes:
            return 0.0
        
        # Should maintain mostly continuous contact (unlike brushing with gaps)
        if active_ratio < self.config.rub_continuity:
            return 0.0
        
        # Check bidirectional balance (not just one direction)
        horizontal_dirs = [d for d in directions if d in ['left', 'right']]
        vertical_dirs = [d for d in directions if d in ['up', 'down']]
        
        # Determine primary axis of rubbing
        if len(horizontal_dirs) >= len(vertical_dirs):
            # Horizontal rubbing
            primary_dirs = horizontal_dirs
            left_count = primary_dirs.count('left')
            right_count = primary_dirs.count('right')
        else:
            # Vertical rubbing
            primary_dirs = vertical_dirs
            left_count = primary_dirs.count('up')    # Treat up as "left"
            right_count = primary_dirs.count('down')  # Treat down as "right"
        
        if len(primary_dirs) == 0:
            return 0.0
        
        # Check balance between both directions
        min_ratio = min(left_count, right_count) / len(primary_dirs)
        
        if min_ratio < self.config.rub_bidirectional_balance:
            return 0.0
        
        # Calculate confidence score
        # More direction changes = more confident it's rubbing
        change_score = min(1.0, direction_changes / 4.0)  # Cap at 4 changes
        
        # Better balance = higher confidence
        balance_score = min_ratio / 0.5  # Perfect balance (0.5) = 1.0
        
        # Continuous contact helps
        continuity_score = min(1.0, active_ratio / 0.8)
        
        # Combined score
        confidence = (
            change_score * 0.4 +
            balance_score * 0.35 +
            continuity_score * 0.25
        )
        
        return confidence
    
    def _check_brushing(self, features: dict) -> float:
        """
        Brushing: Repeated unidirectional strokes with gaps
        
        Modified to distinguish from rubbing:
        - Should NOT have many direction reversals
        - Should have gaps (intermittent)
        """
        directions = features['directions']
        n_activations = features['n_activations']
        direction_changes = features['direction_changes']
        active_ratio = features['active_ratio']
        
        if len(directions) < 2 or n_activations < self.config.brush_min_strokes:
            return 0.0
        
        # If too many direction changes, it's probably rubbing not brushing
        if direction_changes > 2:
            return 0.0
        
        # Brushing should have gaps (intermittent pattern)
        if active_ratio > 0.85:
            return 0.0
        
        # Check direction consistency
        if not directions:
            return 0.0
        
        direction_counts = {}
        for d in directions:
            direction_counts[d] = direction_counts.get(d, 0) + 1
        
        most_common_count = max(direction_counts.values())
        consistency = most_common_count / len(directions)
        
        if consistency < self.config.brush_direction_consistency:
            return 0.0
        
        intermittency_score = min(1.0, features['state_changes'] / (features['n_frames'] * 0.2))
        
        return consistency * 0.7 + intermittency_score * 0.3
    
    def _check_patting(self, features: dict) -> float:
        """Patting: Intermittent coverage with consistent area"""
        n_activations = features['n_activations']
        mean_coverage = features['mean_coverage']
        active_cells = features['active_cells_per_frame']
        
        if n_activations < self.config.pat_min_events:
            return 0.0
        
        if mean_coverage < self.config.pat_min_coverage:
            return 0.0
        
        active_frames = active_cells[active_cells > 0]
        if len(active_frames) < 2:
            return 0.0
        
        area_variance = np.var(active_frames)
        
        if features['active_ratio'] > 0.9:
            return 0.0
        
        if area_variance <= self.config.pat_area_variance_max:
            variance_score = 1.0 - (area_variance / (self.config.pat_area_variance_max + 1))
            intermittency_score = 1.0 - abs(features['active_ratio'] - 0.5) * 2
            
            avg_active = np.mean(active_frames)
            multi_cell_score = min(1.0, avg_active / 2.0)
            
            return variance_score * 0.4 + intermittency_score * 0.3 + multi_cell_score * 0.3
        
        return 0.0
    
    def _check_poking(self, features: dict) -> float:
        """Poking: Single cell repeatedly activated"""
        n_activations = features['n_activations']
        cell_counts = features['cell_activation_count']
        active_cells = features['active_cells_per_frame']
        
        if n_activations < self.config.poke_min_events:
            return 0.0
        
        active_frames = active_cells[active_cells > 0]
        if len(active_frames) == 0:
            return 0.0
        
        avg_cells_when_active = np.mean(active_frames)
        
        if avg_cells_when_active > self.config.poke_max_cells + 0.5:
            return 0.0
        
        total_activations = np.sum(cell_counts)
        if total_activations == 0:
            return 0.0
        
        max_cell_activations = np.max(cell_counts)
        location_consistency = max_cell_activations / total_activations
        
        if location_consistency >= self.config.poke_location_consistency:
            single_cell_score = 1.0 - (avg_cells_when_active - 1.0) / 2.0
            return location_consistency * 0.6 + max(0, single_cell_score) * 0.4
        
        return 0.0
    
    def _check_hugging(self, features: dict) -> float:
        """Hugging: Large area sustained throughout window"""
        coverage_per_frame = features['coverage_per_frame']
        active_ratio = features['active_ratio']
        mean_coverage = features['mean_coverage']
        
        if active_ratio < self.config.hug_stability:
            return 0.0
        
        if mean_coverage < self.config.hug_min_coverage:
            return 0.0
        
        high_coverage_frames = coverage_per_frame >= self.config.hug_min_coverage
        sustained_ratio = np.mean(high_coverage_frames)
        
        if sustained_ratio >= self.config.hug_stability:
            coverage_score = min(1.0, mean_coverage / 0.75)
            stability_score = sustained_ratio
            return coverage_score * 0.5 + stability_score * 0.5
        
        return 0.0


class GestureDetectorNode(Node):
    """ROS2 Node for real-time gesture detection from /fsrFrames"""
    
    def __init__(self):
        super().__init__('gesture_detector')
        
        # Parameters
        self.declare_parameter('window_duration', 2.0)
        self.declare_parameter('binary_threshold', 1)
        self.declare_parameter('debug', False)
        
        window_duration = self.get_parameter('window_duration').value
        binary_threshold = self.get_parameter('binary_threshold').value
        self.debug = self.get_parameter('debug').value
        
        # Config
        self.config = GestureConfig(
            window_duration=window_duration,
            sample_rate=15.0,
            binary_threshold=binary_threshold
        )
        self.analyzer = GestureAnalyzer(self.config)
        
        # Buffer
        self.buffer_size = int(window_duration * self.config.sample_rate)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.timestamp_buffer = deque(maxlen=self.buffer_size)
        
        # ROS interfaces
        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/fsrFrames',
            self.fsr_callback,
            10
        )
        
        self.gesture_pub = self.create_publisher(String, '/detected_gesture', 10)
        
        # Analysis at 5 Hz
        self.analysis_timer = self.create_timer(0.2, self.analyze_gestures)
        
        # State tracking
        self.last_gesture = GestureType.NONE
        self.gesture_hold_count = 0
        self.min_hold_frames = 2
        
        self.get_logger().info(
            f'Gesture Detector Started\n'
            f'  - Supported gestures: {[g.value for g in GestureType if g != GestureType.NONE]}\n'
            f'  - Buffer: {self.buffer_size} frames ({window_duration}s @ 15 FPS)'
        )
    
    def fsr_callback(self, msg: Int32MultiArray):
        """Handle incoming FSR matrix data"""
        try:
            raw_data = np.array(msg.data, dtype=np.int32)
            
            if len(raw_data) != 8:
                self.get_logger().warn(f'Unexpected data length: {len(raw_data)}')
                return
            
            raw_matrix = raw_data.reshape(2, 4)
            binary_matrix = (raw_matrix >= self.config.binary_threshold).astype(np.int8)
            timestamp = self.get_clock().now().nanoseconds / 1e9
            
            self.frame_buffer.append(binary_matrix)
            self.timestamp_buffer.append(timestamp)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def analyze_gestures(self):
        """Periodic gesture analysis"""
        if len(self.frame_buffer) < 10:
            return
        
        frames = list(self.frame_buffer)
        timestamps = list(self.timestamp_buffer)
        
        gesture, confidence, debug_info = self.analyzer.analyze(frames, timestamps)
        
        # Temporal smoothing
        if gesture == self.last_gesture and gesture != GestureType.NONE:
            self.gesture_hold_count += 1
        elif gesture != GestureType.NONE:
            self.gesture_hold_count = 1
            self.last_gesture = gesture
        else:
            self.gesture_hold_count = max(0, self.gesture_hold_count - 1)
            if self.gesture_hold_count == 0:
                self.last_gesture = GestureType.NONE
        
        # Publish
        if self.gesture_hold_count >= self.min_hold_frames:
            msg = String()
            msg.data = f'{gesture.value}:{confidence:.2f}'
            self.gesture_pub.publish(msg)
            
            self.get_logger().info(
                f'DETECTED: {gesture.value.upper()} (confidence: {confidence:.2f})'
            )
        
        if self.debug:
            self.get_logger().info(
                f'Scores: {debug_info.get("all_scores", {})}\n'
                f'Dir changes: {debug_info.get("direction_changes", 0)}, '
                f'Active: {debug_info.get("active_ratio", 0):.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = GestureDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()