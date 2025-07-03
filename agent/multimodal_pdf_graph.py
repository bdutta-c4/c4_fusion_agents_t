#!/usr/bin/env python3
"""
LangGraph implementation of Enhanced Multimodal PDF Parser
Uses send operations for parallel slide processing and multi-level summarization
"""

import os
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Annotated
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import operator

# LangGraph imports
from langgraph.graph import StateGraph, START, END, Send
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Core PDF processing
import fitz  # PyMuPDF
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SlideTask:
    """Individual slide processing task"""
    slide_number: int
    image_path: str
    file_stem: str

@dataclass
class SlideAnalysis:
    """Container for individual slide analysis"""
    slide_number: int
    slide_title: str
    text_content: str
    tables: List[Dict]
    charts: List[Dict]
    key_insights: List[str]
    facts_and_metrics: List[str]
    red_flags: List[str]
    slide_summary: str
    raw_analysis: str
    image_path: str

class PDFAnalysisState(TypedDict):
    """State for PDF analysis workflow"""
    pdf_path: str
    output_dir: str
    total_slides: int
    slide_tasks: List[SlideTask]
    slide_analyses: List[SlideAnalysis]
    page_by_page_summary: str
    executive_summary: str
    key_facts_metrics: List[str]
    critical_insights: List[str]
    red_flags: List[str]
    recommendations: List[str]
    presentation_title: str
    metadata: Dict[str, Any]
    messages: Annotated[List[BaseMessage], add_messages]

class MultimodalPDFGraph:
    """LangGraph implementation of multimodal PDF parser"""
    
    def __init__(self, output_dir: str = "langgraph_analysis", openai_api_key: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(PDFAnalysisState)
        
        # Add nodes
        workflow.add_node("extract_slides", self.extract_slides)
        workflow.add_node("analyze_slide", self.analyze_single_slide)
        workflow.add_node("collect_analyses", self.collect_slide_analyses)
        workflow.add_node("create_page_summary", self.create_page_by_page_summary)
        workflow.add_node("create_executive_summary", self.create_executive_summary)
        workflow.add_node("save_results", self.save_analysis_results)
        
        # Define workflow edges
        workflow.add_edge(START, "extract_slides")
        workflow.add_conditional_edges(
            "extract_slides",
            self.route_to_slide_analysis,
            ["analyze_slide"]
        )
        workflow.add_edge("analyze_slide", "collect_analyses")
        workflow.add_edge("collect_analyses", "create_page_summary")
        workflow.add_edge("create_page_summary", "create_executive_summary")
        workflow.add_edge("create_executive_summary", "save_results")
        workflow.add_edge("save_results", END)
        
        return workflow.compile()
    
    def extract_slides(self, state: PDFAnalysisState) -> PDFAnalysisState:
        """Extract slides from PDF and prepare for parallel processing"""
        logger.info(f"Extracting slides from: {state['pdf_path']}")
        
        pdf_path = Path(state['pdf_path'])
        doc = fitz.open(str(pdf_path))
        slide_tasks = []
        
        for slide_num in range(len(doc)):
            page = doc[slide_num]
            
            # Extract slide image
            image_path = self._extract_slide_image(page, slide_num, pdf_path.stem)
            
            slide_tasks.append(SlideTask(
                slide_number=slide_num + 1,
                image_path=image_path,
                file_stem=pdf_path.stem
            ))
        
        doc.close()
        
        state.update({
            "total_slides": len(slide_tasks),
            "slide_tasks": slide_tasks,
            "slide_analyses": [],
            "messages": [HumanMessage(content=f"Extracted {len(slide_tasks)} slides for analysis")]
        })
        
        return state
    
    def route_to_slide_analysis(self, state: PDFAnalysisState) -> List[Send]:
        """Route each slide to parallel analysis using Send operations"""
        logger.info(f"Routing {len(state['slide_tasks'])} slides for parallel analysis")
        
        # Create Send operations for parallel processing
        sends = []
        for task in state['slide_tasks']:
            # Create individual state for each slide
            slide_state = {
                **state,
                "current_slide_task": task,
                "messages": [HumanMessage(content=f"Analyzing slide {task.slide_number}")]
            }
            sends.append(Send("analyze_slide", slide_state))
        
        return sends
    
    def analyze_single_slide(self, state: PDFAnalysisState) -> SlideAnalysis:
        """Analyze a single slide using GPT-4 Vision"""
        task = state["current_slide_task"]
        logger.info(f"Analyzing slide {task.slide_number}")
        
        try:
            # Encode image to base64
            with open(task.image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # GPT-4 Vision analysis
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._get_slide_analysis_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    analysis_data = json.loads(json_str)
                else:
                    analysis_data = self._create_fallback_analysis(content, task.slide_number)
            except json.JSONDecodeError:
                analysis_data = self._create_fallback_analysis(content, task.slide_number)
            
            # Create SlideAnalysis object
            return SlideAnalysis(
                slide_number=task.slide_number,
                slide_title=analysis_data.get('slide_title', f'Slide {task.slide_number}'),
                text_content=analysis_data.get('text_content', ''),
                tables=analysis_data.get('tables', []),
                charts=analysis_data.get('charts', []),
                key_insights=analysis_data.get('key_insights', []),
                facts_and_metrics=analysis_data.get('facts_and_metrics', []),
                red_flags=analysis_data.get('red_flags', []),
                slide_summary=analysis_data.get('slide_summary', ''),
                raw_analysis=json.dumps(analysis_data, indent=2),
                image_path=task.image_path
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze slide {task.slide_number}: {str(e)}")
            return self._create_error_analysis(task.slide_number, str(e), task.image_path)
    
    def collect_slide_analyses(self, state: PDFAnalysisState) -> PDFAnalysisState:
        """Collect all slide analyses and sort by slide number"""
        # This node receives results from parallel slide analysis
        # Sort analyses by slide number to maintain order
        if "slide_analyses" not in state:
            state["slide_analyses"] = []
        
        # Sort by slide number
        state["slide_analyses"].sort(key=lambda x: x.slide_number)
        
        logger.info(f"Collected {len(state['slide_analyses'])} slide analyses")
        
        state["messages"].append(
            AIMessage(content=f"Completed analysis of {len(state['slide_analyses'])} slides")
        )
        
        return state
    
    def create_page_by_page_summary(self, state: PDFAnalysisState) -> PDFAnalysisState:
        """Create flowing page-by-page summary"""
        logger.info("Creating page-by-page summary")
        
        # Prepare slide summaries for GPT
        slides_text = ""
        for slide in state["slide_analyses"]:
            slides_text += f"\n--- Slide {slide.slide_number}: {slide.slide_title} ---\n"
            slides_text += f"Summary: {slide.slide_summary}\n"
            slides_text += f"Key Insights: {', '.join(slide.key_insights)}\n"
            slides_text += f"Facts & Metrics: {', '.join(slide.facts_and_metrics)}\n"
            if slide.red_flags:
                slides_text += f"Red Flags: {', '.join(slide.red_flags)}\n"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_page_summary_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Create a page-by-page summary for this presentation:\n{slides_text}"
                    }
                ],
                max_tokens=2000,
                temperature=0.2
            )
            
            state["page_by_page_summary"] = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to create page-by-page summary: {str(e)}")
            state["page_by_page_summary"] = "Page-by-page summary generation failed."
        
        state["messages"].append(AIMessage(content="Created page-by-page summary"))
        return state
    
    def create_executive_summary(self, state: PDFAnalysisState) -> PDFAnalysisState:
        """Create executive summary with key insights"""
        logger.info("Creating executive summary")
        
        # Compile all data for executive summary
        all_facts = []
        all_insights = []
        all_red_flags = []
        
        for slide in state["slide_analyses"]:
            all_facts.extend(slide.facts_and_metrics)
            all_insights.extend(slide.key_insights)
            all_red_flags.extend(slide.red_flags)
        
        summary_data = f"""
Page-by-Page Summary:
{state['page_by_page_summary']}

All Facts & Metrics:
{chr(10).join(f"• {fact}" for fact in all_facts)}

All Key Insights:
{chr(10).join(f"• {insight}" for insight in all_insights)}

Red Flags:
{chr(10).join(f"• {flag}" for flag in all_red_flags)}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_executive_summary_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Create an executive summary based on this presentation analysis:\n{summary_data}"
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    executive_data = json.loads(json_str)
                else:
                    executive_data = self._create_fallback_executive_summary(content)
            except json.JSONDecodeError:
                executive_data = self._create_fallback_executive_summary(content)
            
            state.update({
                "executive_summary": executive_data.get('executive_summary', ''),
                "key_facts_metrics": executive_data.get('key_facts_metrics', []),
                "critical_insights": executive_data.get('critical_insights', []),
                "red_flags": executive_data.get('red_flags', []),
                "recommendations": executive_data.get('recommendations', [])
            })
            
        except Exception as e:
            logger.error(f"Failed to create executive summary: {str(e)}")
            state.update({
                "executive_summary": f"Executive summary generation failed: {str(e)}",
                "key_facts_metrics": [],
                "critical_insights": [],
                "red_flags": [],
                "recommendations": []
            })
        
        state["messages"].append(AIMessage(content="Created executive summary with key insights"))
        return state
    
    def save_analysis_results(self, state: PDFAnalysisState) -> PDFAnalysisState:
        """Save comprehensive analysis results"""
        logger.info("Saving analysis results")
        
        file_stem = Path(state["pdf_path"]).stem
        
        # Extract presentation title
        presentation_title = "Presentation Analysis"
        if state["slide_analyses"] and state["slide_analyses"][0].slide_title:
            presentation_title = state["slide_analyses"][0].slide_title
        
        # Create complete summary object
        presentation_summary = {
            "file_path": state["pdf_path"],
            "presentation_title": presentation_title,
            "total_slides": state["total_slides"],
            "slide_analyses": [asdict(slide) for slide in state["slide_analyses"]],
            "page_by_page_summary": state["page_by_page_summary"],
            "executive_summary": state["executive_summary"],
            "key_facts_metrics": state["key_facts_metrics"],
            "critical_insights": state["critical_insights"],
            "red_flags": state["red_flags"],
            "recommendations": state["recommendations"],
            "metadata": {
                "file_name": Path(state["pdf_path"]).name,
                "total_slides": state["total_slides"],
                "analysis_timestamp": datetime.now().isoformat(),
                "extraction_method": "LangGraph + GPT-4 Vision + Parallel Processing"
            },
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # Save complete analysis as JSON
        json_file = self.output_dir / "reports" / f"{file_stem}_langgraph_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(presentation_summary, f, indent=2, ensure_ascii=False)
        
        # Save executive summary as Markdown
        exec_file = self.output_dir / "summaries" / f"{file_stem}_executive_summary.md"
        self._save_executive_summary_md(presentation_summary, exec_file)
        
        state["messages"].append(AIMessage(content=f"Analysis results saved to {self.output_dir}"))
        logger.info(f"Analysis saved to {self.output_dir}")
        
        return state
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main entry point for PDF analysis"""
        initial_state = PDFAnalysisState(
            pdf_path=pdf_path,
            output_dir=str(self.output_dir),
            total_slides=0,
            slide_tasks=[],
            slide_analyses=[],
            page_by_page_summary="",
            executive_summary="",
            key_facts_metrics=[],
            critical_insights=[],
            red_flags=[],
            recommendations=[],
            presentation_title="",
            metadata={},
            messages=[]
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    # Helper methods (abbreviated for space)
    def _extract_slide_image(self, page, slide_num: int, file_stem: str) -> str:
        """Extract slide as high-quality image"""
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img_filename = f"{file_stem}_slide_{slide_num + 1}.png"
        img_path = self.output_dir / "images" / img_filename
        pix.save(str(img_path))
        return str(img_path)
    
    def _get_slide_analysis_prompt(self) -> str:
        """Comprehensive slide analysis prompt"""
        return """You are an expert business analyst examining a PowerPoint presentation slide. 
        Analyze this slide comprehensively and extract structured information in JSON format.
        
        Return your analysis in this JSON structure:
        {
            "slide_title": "Main title of the slide",
            "text_content": "All text content found on the slide",
            "tables": [],
            "charts": [],
            "key_insights": ["insight1", "insight2"],
            "facts_and_metrics": ["fact1", "fact2"],
            "red_flags": ["flag1", "flag2"],
            "slide_summary": "2-3 sentence summary"
        }"""
    
    def _get_page_summary_prompt(self) -> str:
        """Prompt for creating page-by-page summary"""
        return """Create a flowing narrative that follows the logical progression of the presentation,
        highlights key points from each slide, and connects themes across slides."""
    
    def _get_executive_summary_prompt(self) -> str:
        """Prompt for creating executive summary"""
        return """Create an executive summary for a busy account manager who needs to understand 
        the key points within 30 seconds. Format as JSON with executive_summary, key_facts_metrics, 
        critical_insights, red_flags, and recommendations."""
    
    def _create_fallback_analysis(self, content: str, slide_num: int) -> Dict:
        """Create fallback analysis structure"""
        return {
            "slide_title": f"Slide {slide_num}",
            "text_content": content,
            "tables": [],
            "charts": [],
            "key_insights": [content[:200] + "..." if len(content) > 200 else content],
            "facts_and_metrics": [],
            "red_flags": [],
            "slide_summary": content[:100] + "..." if len(content) > 100 else content
        }
    
    def _create_error_analysis(self, slide_num: int, error: str, image_path: str) -> SlideAnalysis:
        """Create error analysis object"""
        return SlideAnalysis(
            slide_number=slide_num,
            slide_title=f"Slide {slide_num} (Analysis Failed)",
            text_content=f"Analysis failed: {error}",
            tables=[], charts=[], key_insights=[], facts_and_metrics=[],
            red_flags=[], slide_summary=f"Analysis failed for slide {slide_num}",
            raw_analysis=f"Error: {error}", image_path=image_path
        )
    
    def _create_fallback_executive_summary(self, content: str) -> Dict:
        """Create fallback executive summary"""
        return {
            "executive_summary": content[:300] + "..." if len(content) > 300 else content,
            "key_facts_metrics": [], "critical_insights": [],
            "red_flags": [], "recommendations": []
        }
    
    def _save_executive_summary_md(self, summary: Dict, file_path: Path):
        """Save executive summary as Markdown"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Executive Summary: {summary['presentation_title']}\n\n")
            f.write(f"**File:** {Path(summary['file_path']).name}\n")
            f.write(f"**Slides:** {summary['total_slides']}\n")
            f.write(f"**Analyzed:** {summary['extraction_timestamp']}\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"{summary['executive_summary']}\n\n")
            
            if summary['key_facts_metrics']:
                f.write("## Key Facts & Metrics\n\n")
                for fact in summary['key_facts_metrics']:
                    f.write(f"• {fact}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph PDF analysis with parallel processing")
    parser.add_argument("pdf_path", help="Path to PDF presentation file")
    parser.add_argument("-o", "--output", default="langgraph_analysis", help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MultimodalPDFGraph(args.output, args.api_key)
    
    # Analyze presentation
    result = analyzer.analyze_pdf(args.pdf_path)
    
    print(f"\n✅ LangGraph Analysis Complete!")
    print(f"📊 Processed {result['total_slides']} slides in parallel")
    print(f"📁 Results saved to: {args.output}/")
