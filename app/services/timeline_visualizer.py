"""
Timeline visualization for phases and events
"""
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from io import BytesIO
import base64


class TimelineVisualizer:
    """Create beautiful timeline visualizations"""
    
    # Phase colors (consistent across app)
    PHASE_COLORS = {
        'Portal Placement': '#9C27B0',
        'Diagnostic Arthroscopy': '#3F51B5',
        'Labral Mobilization': '#2196F3',
        'Glenoid Preparation': '#00BCD4',
        'Anchor Placement': '#4CAF50',
        'Suture Passage': '#FFC107',
        'Suture Tensioning': '#FF9800',
        'Final Inspection': '#795548',
        'Suture Tensioning': '#FF9800',
        'Diagnositc Arthroscopy': '#3F51B5',  # Typo variant
        'Diagnostic Arthroscopy': '#3F51B5',
        'Labral Mobilization': '#2196F3'
    }
    
    # Event markers
    EVENT_STYLES = {
        'Bleeding': {'marker': 'v', 'color': '#F44336', 'size': 120},
        'Suture Attempt': {'marker': '^', 'color': '#4CAF50', 'size': 120},
        'Portal Placement': {'marker': 'o', 'color': '#9C27B0', 'size': 80},
        'Instruments': {'marker': 's', 'color': '#607D8B', 'size': 100}
    }
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def create_case_timeline(self, case_id, output_path=None, return_base64=False):
        """
        Create timeline visualization for a case
        
        Args:
            case_id: Case ID to visualize
            output_path: Path to save image (optional)
            return_base64: If True, return base64 encoded image string
            
        Returns:
            Path to saved image, base64 string, or BytesIO buffer
        """
        from app.models.database import Case
        
        session = self.db.get_session()
        
        try:
            case = session.query(Case).filter_by(case_id=case_id).first()
            if not case:
                return None
            
            # Create figure with larger size to prevent text cutoff
            fig = plt.figure(figsize=(16, 8))
            
            # Main timeline axis (phases)
            ax_phases = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            # Events axis
            ax_events = plt.subplot2grid((3, 1), (2, 0))
            
            total_frames = case.total_frames if case.total_frames else 1
            total_minutes = case.actual_duration_min if case.actual_duration_min else 60
            
            # Draw phases
            y_position = 0
            phase_names_shown = set()
            
            for phase in sorted(case.phases, key=lambda p: p.start_frame):
                start_pct = phase.start_frame / total_frames
                width_pct = (phase.end_frame - phase.start_frame) / total_frames
                
                # Get color
                base_name = phase.phase_name.replace(f" (Anchor #{phase.anchor_number})", "") if phase.anchor_number else phase.phase_name
                color = self.PHASE_COLORS.get(base_name, '#999999')
                
                # Draw bar
                rect = Rectangle((start_pct, y_position), width_pct, 0.8,
                                facecolor=color, edgecolor='white', linewidth=1)
                ax_phases.add_patch(rect)
                
                # Add label (avoid duplicates)
                label = f"{base_name}"
                if phase.anchor_number:
                    label += f" #{phase.anchor_number}"
                
                if len(label) < 30 and width_pct > 0.05:  # Only if bar is wide enough
                    ax_phases.text(start_pct + width_pct/2, y_position + 0.4, label,
                                  ha='center', va='center', fontsize=6, 
                                  fontweight='bold', color='white')
            
            # Format phase axis
            ax_phases.set_ylim(-0.2, 1.2)
            ax_phases.set_xlim(0, 1)
            ax_phases.set_yticks([])
            ax_phases.set_ylabel('Surgical Phases', fontsize=10, fontweight='bold')
            ax_phases.set_title(f'Case {case_id} - Timeline Visualization', 
                               fontsize=12, fontweight='bold', pad=15)
            ax_phases.set_xticks([])
            ax_phases.spines['top'].set_visible(False)
            ax_phases.spines['right'].set_visible(False)
            ax_phases.spines['left'].set_visible(False)
            
            # Draw events on separate axis
            event_types_shown = {}
            
            for event in case.events:
                pos_pct = event.event_frame / total_frames
                event_type = event.event_type
                
                style = self.EVENT_STYLES.get(event_type, 
                    {'marker': 'o', 'color': '#999999', 'size': 80})
                
                # Track Y position for each event type
                if event_type not in event_types_shown:
                    event_types_shown[event_type] = len(event_types_shown) * 0.25
                
                y_pos = event_types_shown[event_type]
                
                # Draw marker
                ax_events.scatter(pos_pct, y_pos, marker=style['marker'], 
                                 color=style['color'], s=style['size'],
                                 edgecolors='white', linewidths=1.5, zorder=10, alpha=0.9)
                
                # Add severity/outcome annotation for important events
                if event.severity == 'Severe' or (event.outcome == 'Fail' and event_type == 'Suture Attempt'):
                    ax_events.annotate('!', xy=(pos_pct, y_pos),
                                      xytext=(0, 10), textcoords='offset points',
                                      fontsize=12, color='red', fontweight='bold',
                                      ha='center')
            
            # Format events axis
            ax_events.set_ylim(-0.3, max(1.0, len(event_types_shown) * 0.25 + 0.3))
            ax_events.set_xlim(0, 1)
            ax_events.set_xlabel('Time (minutes)', fontsize=10, fontweight='bold')
            ax_events.set_ylabel('Events', fontsize=10, fontweight='bold')
            
            # X-axis time labels
            time_points = np.linspace(0, 1, 11)
            time_labels = [f"{int(total_minutes * t)}" for t in time_points]
            ax_events.set_xticks(time_points)
            ax_events.set_xticklabels(time_labels, fontsize=8)
            
            # Y-axis event type labels
            if event_types_shown:
                ax_events.set_yticks([event_types_shown[et] for et in event_types_shown.keys()])
                ax_events.set_yticklabels(list(event_types_shown.keys()), fontsize=8)
            else:
                ax_events.set_yticks([])
            
            ax_events.spines['top'].set_visible(False)
            ax_events.spines['right'].set_visible(False)
            ax_events.grid(True, axis='x', alpha=0.3, linestyle='--')
            
            # Add legend
            legend_elements = []
            
            # Phase legend
            unique_phases = set()
            for phase in case.phases:
                base_name = phase.phase_name.replace(f" (Anchor #{phase.anchor_number})", "") if phase.anchor_number else phase.phase_name
                if base_name not in unique_phases:
                    unique_phases.add(base_name)
                    color = self.PHASE_COLORS.get(base_name, '#999999')
                    legend_elements.append(mpatches.Patch(color=color, label=base_name))
            
            # Event legend
            for event_type in event_types_shown.keys():
                style = self.EVENT_STYLES.get(event_type, 
                    {'marker': 'o', 'color': '#999999', 'size': 80})
                legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], 
                                                  color='w', markerfacecolor=style['color'],
                                                  markersize=8, label=event_type))
            
            # Place legend outside
            if legend_elements:
                ax_phases.legend(handles=legend_elements, loc='center left', 
                               bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
            
            plt.tight_layout()
            
            # Save or return
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"✅ Timeline saved to {output_path}")
                return output_path
            
            if return_base64:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                return f"data:image/png;base64,{img_base64}"
            
            # Return BytesIO buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
            
        except Exception as e:
            print(f"❌ Error creating timeline: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            session.close()

