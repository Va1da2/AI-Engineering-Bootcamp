from langsmith import Client as LSClient


def submit_feedback(
    trace_id: str,
    langsmith_client: LSClient,
    feedback_score: int = None,
    feedback_text: str = "",
    feedback_source_type: str = "api"):

    
    if feedback_score and feedback_text:
        langsmith_client.create_feedback(
            run_id=trace_id,
            key="thumbs_and_comment",
            score=feedback_score,
            value=feedback_text,
            feedback_source_type=feedback_source_type
        )
    
    elif feedback_score:
        langsmith_client.create_feedback(
            run_id=trace_id,
            key="thumbs",
            score=feedback_score,
            feedback_source_type=feedback_source_type
        )
    
    elif feedback_text:
        langsmith_client.create_feedback(
            run_id=trace_id,
            key="comment",
            value=feedback_text,
            feedback_source_type=feedback_source_type
        )