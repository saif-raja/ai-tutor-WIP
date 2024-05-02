# db_setup.py
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

# Setup the base class for declarative class definitions.
Base = declarative_base()

# Define the ConversationState model.
class ConversationState(Base):
    __tablename__ = 'conversation_states'
    session_id = Column(String, primary_key=True)
    state = Column(Text)

    def __repr__(self):
        return f"<ConversationState(session_id='{self.session_id}', state='{self.state}')>"

# Global variable for the engine
engine = create_engine('sqlite:///conversation_states.db', echo=True)

# Session factory, configured from the engine
Session = scoped_session(sessionmaker(bind=engine))

def init_db():
    """Creates all tables in the database (if they don't exist) and initializes the DB."""
    Base.metadata.create_all(engine)


class DatabaseMemory:
    def __init__(self):
        self.session = Session()

    def get(self, session_id):
        # Retrieve the conversation state from the database
        state = self.session.query(ConversationState).filter_by(session_id=session_id).first()
        if state is not None:
            return state.state
        return None

    def set(self, session_id, value):
        # Store the conversation state to the database
        state = self.session.query(ConversationState).filter_by(session_id=session_id).first()
        if state is None:
            state = ConversationState(session_id=session_id, state=value)
            self.session.add(state)
        else:
            state.state = value
        self.session.commit()

    def __del__(self):
        self.session.close()


        
if __name__ == '__main__':
    init_db()