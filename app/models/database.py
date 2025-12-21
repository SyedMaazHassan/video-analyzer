"""
Database models and manager for Surgical Analysis Platform
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()


class Surgeon(Base):
    """Surgeon information"""
    __tablename__ = 'surgeons'
    
    surgeon_id = Column(String(50), primary_key=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    department = Column(String(100))
    specialty = Column(String(100))
    email = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    cases = relationship('Case', back_populates='surgeon', cascade='all, delete-orphan')
    
    @property
    def full_name(self):
        return f"Dr. {self.first_name} {self.last_name}"
    
    def __repr__(self):
        return f"<Surgeon {self.surgeon_id}: {self.full_name}>"


class Case(Base):
    """Individual surgical case"""
    __tablename__ = 'cases'
    
    case_id = Column(String(50), primary_key=True)
    surgeon_id = Column(String(50), ForeignKey('surgeons.surgeon_id'))
    procedure_type = Column(String(100))
    procedure_date = Column(DateTime)
    video_path = Column(Text)
    video_duration_sec = Column(Float)
    video_fps = Column(Integer)
    total_frames = Column(Integer)
    estimated_duration_min = Column(Float)
    actual_duration_min = Column(Float)
    processing_status = Column(String(20))  # queued, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    surgeon = relationship('Surgeon', back_populates='cases')
    phases = relationship('Phase', back_populates='case', cascade='all, delete-orphan')
    events = relationship('Event', back_populates='case', cascade='all, delete-orphan')
    resources = relationship('Resource', back_populates='case', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Case {self.case_id}: {self.surgeon.full_name if self.surgeon else 'N/A'}>"


class Phase(Base):
    """Surgical phase within a case"""
    __tablename__ = 'phases'
    
    phase_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey('cases.case_id'))
    phase_name = Column(String(100))
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    duration_sec = Column(Float)
    anchor_number = Column(Integer, nullable=True)
    confidence_score = Column(Float)
    idle_time_sec = Column(Float, default=0.0)
    
    # Relationships
    case = relationship('Case', back_populates='phases')
    
    def __repr__(self):
        return f"<Phase {self.phase_name} in {self.case_id}>"


class Event(Base):
    """Surgical event within a case"""
    __tablename__ = 'events'
    
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey('cases.case_id'))
    event_type = Column(String(50))  # bleeding, suture_attempt, portal_placement, instrument
    event_frame = Column(Integer)
    event_time_sec = Column(Float)
    
    # Event-specific metadata
    anchor_number = Column(Integer, nullable=True)
    attempt_number = Column(Integer, nullable=True)
    outcome = Column(String(20), nullable=True)  # success, fail
    severity = Column(String(20), nullable=True)  # mild, moderate, severe
    instrument_type = Column(String(100), nullable=True)
    
    confidence_score = Column(Float)
    notes = Column(Text)
    
    # Relationships
    case = relationship('Case', back_populates='events')
    
    def __repr__(self):
        return f"<Event {self.event_type} at {self.event_time_sec}s in {self.case_id}>"


class Resource(Base):
    """Resource utilization for a case"""
    __tablename__ = 'resources'
    
    resource_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey('cases.case_id'))
    implants_count = Column(Integer)
    disposables_count = Column(Integer)
    electrocautery_usage_percent = Column(Float)
    anchor_repositions = Column(Integer)
    
    # Relationships
    case = relationship('Case', back_populates='resources')
    
    def __repr__(self):
        return f"<Resources for {self.case_id}>"


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, db_path='data/surgical_analysis.db'):
        """Initialize database manager"""
        # Ensure data directory exists
        data_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"📁 Created data directory: {data_dir}")

        # Check if database file exists (for logging)
        db_exists = os.path.exists(db_path)

        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')

        # Create all tables (safe to call even if they exist)
        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine)

        if db_exists:
            print(f"✅ Database loaded from {db_path}")
        else:
            print(f"✅ New database created at {db_path}")
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def init_sample_surgeons(self):
        """Initialize with sample surgeons"""
        session = self.get_session()
        
        try:
            # Check if surgeons already exist
            if session.query(Surgeon).count() > 0:
                print("✅ Surgeons already exist in database")
                return
            
            surgeons = [
                Surgeon(
                    surgeon_id='S001',
                    first_name='Sarah',
                    last_name='Anderson',
                    department='Orthopedics',
                    specialty='Sports Medicine',
                    email='s.anderson@hospital.com'
                ),
                Surgeon(
                    surgeon_id='S002',
                    first_name='Michael',
                    last_name='Johnson',
                    department='Orthopedics',
                    specialty='Sports Medicine',
                    email='m.johnson@hospital.com'
                ),
                Surgeon(
                    surgeon_id='S003',
                    first_name='Emily',
                    last_name='Smith',
                    department='Orthopedics',
                    specialty='Sports Medicine',
                    email='e.smith@hospital.com'
                ),
            ]
            
            session.add_all(surgeons)
            session.commit()
            print(f"✅ Created {len(surgeons)} sample surgeons")
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error creating surgeons: {e}")
        finally:
            session.close()
    
    def clear_all_data(self):
        """Clear all data from database (for testing)"""
        session = self.get_session()
        try:
            session.query(Resource).delete()
            session.query(Event).delete()
            session.query(Phase).delete()
            session.query(Case).delete()
            session.query(Surgeon).delete()
            session.commit()
            print("✅ Cleared all data from database")
        except Exception as e:
            session.rollback()
            print(f"❌ Error clearing data: {e}")
        finally:
            session.close()



