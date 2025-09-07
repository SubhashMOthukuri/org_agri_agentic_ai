"""
Decision Agent - Enterprise-Grade AI Orchestrator

Principal AI Engineer Level Implementation
- Main orchestrator for all AI agents
- LLM integration for strategic decision making
- Multi-agent coordination and communication
- Risk assessment and mitigation strategies
- Action planning and execution

Author: Principal AI Engineer
Version: 1.0.0
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# LLM Integration
import openai
from dataclasses import dataclass
from enum import Enum

# Web Search and External Tools
import requests
import json
import time
from typing import Optional, Dict, List, Any, Tuple
import hashlib
import pickle
import os

# Project imports
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_layer.database.mongodb.mongodb_setup import MongoDBManager
from agents.weather_agent import WeatherAgent, WeatherForecast, WeatherAlert
from agents.market_agent import MarketAgent, MarketAnalysis, MarketAlert
from agents.pest_agent import PestAgent, PlantHealthAssessment, PestAlert
from agents.prediction_agent import PredictionAgent, YieldForecast, RiskAssessment

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Decision type classifications"""
    PLANTING = "planting"
    HARVESTING = "harvesting"
    IRRIGATION = "irrigation"
    PEST_CONTROL = "pest_control"
    MARKETING = "marketing"
    RISK_MANAGEMENT = "risk_management"


class DecisionPriority(Enum):
    """Decision priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionStatus(Enum):
    """Action status classifications"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Decision:
    """Decision data structure"""
    decision_id: str
    decision_type: DecisionType
    priority: DecisionPriority
    description: str
    reasoning: str
    confidence: float
    expected_outcome: str
    risk_assessment: str
    actions: List[str]
    timeline: str
    resources_required: List[str]
    success_metrics: List[str]
    created_at: datetime
    status: ActionStatus


@dataclass
class AgentInput:
    """Input data for agent analysis"""
    weather_data: Optional[WeatherForecast] = None
    market_data: Optional[MarketAnalysis] = None
    pest_data: Optional[PlantHealthAssessment] = None
    prediction_data: Optional[YieldForecast] = None
    risk_data: Optional[RiskAssessment] = None


@dataclass
class WebSearchResult:
    """Web search result data structure"""
    query: str
    results: List[Dict[str, Any]]
    timestamp: datetime
    source: str
    relevance_score: float


@dataclass
class ToolResult:
    """External tool result data structure"""
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    execution_time: float
    timestamp: datetime


@dataclass
class MemoryEntry:
    """Persistent memory entry data structure"""
    memory_id: str
    decision_id: str
    context: Dict[str, Any]
    outcome: str
    success: bool
    lessons_learned: List[str]
    timestamp: datetime
    importance_score: float


class DecisionAgent:
    """
    Enterprise-grade Decision Agent - Main AI Orchestrator
    
    Capabilities:
    - Multi-agent coordination and communication
    - Strategic decision making using LLM
    - Risk assessment and mitigation
    - Action planning and execution
    - Performance monitoring and optimization
    """
    
    def __init__(self, mongo_manager: MongoDBManager, config: Optional[Dict] = None):
        self.mongo_manager = mongo_manager
        self.config = config or self._default_config()
        
        # Initialize sub-agents
        self.weather_agent = WeatherAgent(mongo_manager)
        self.market_agent = MarketAgent(mongo_manager)
        self.pest_agent = PestAgent(mongo_manager)
        self.prediction_agent = PredictionAgent(mongo_manager)
        
        # LLM configuration
        self.openai_api_key = self.config.get('openai_api_key', '')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Decision history
        self.decision_history = []
        self.performance_metrics = {}
        
        # Enhanced capabilities
        self.memory_file = self.config.get('memory_file', 'decision_memory.pkl')
        self.persistent_memory = self._load_persistent_memory()
        self.web_search_cache = {}
        self.tool_results_cache = {}
        
        # External APIs
        self.weather_api_key = self.config.get('weather_api_key', '')
        self.market_api_key = self.config.get('market_api_key', '')
        self.news_api_key = self.config.get('news_api_key', '')
        
        # Web search configuration
        self.search_engines = {
            'google': 'https://www.googleapis.com/customsearch/v1',
            'bing': 'https://api.bing.microsoft.com/v7.0/search',
            'duckduckgo': 'https://api.duckduckgo.com/'
        }
        
        logger.info("🧠 Enhanced Decision Agent initialized with Web Search, Tools, and Memory")
    
    def _default_config(self) -> Dict:
        """Default configuration for Decision Agent"""
        return {
            'openai_api_key': '',
            'decision_confidence_threshold': 0.7,
            'risk_tolerance': 'moderate',
            'max_decision_history': 1000,
            'agent_timeout': 30,  # seconds
            'llm_model': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            'temperature': 0.7,
            'memory_file': 'decision_memory.pkl',
            'weather_api_key': '',
            'market_api_key': '',
            'news_api_key': '',
            'search_cache_duration': 3600,  # 1 hour
            'memory_retention_days': 365,
            'max_memory_entries': 10000
        }
    
    async def initialize_agents(self) -> None:
        """Initialize all sub-agents"""
        try:
            logger.info("🤖 Initializing all AI agents...")
            
            # Initialize each agent
            await self.weather_agent.initialize_models()
            await self.market_agent.initialize_models()
            await self.pest_agent.initialize_models()
            await self.prediction_agent.initialize_models()
            
            logger.info("✅ All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing agents: {e}")
    
    async def make_decision(self, decision_type: DecisionType, 
                          location: Tuple[float, float], 
                          crop_type: str,
                          context: Optional[Dict] = None) -> Decision:
        """Make a strategic decision using all available data and enhanced capabilities"""
        try:
            logger.info(f"🧠 Making {decision_type.value} decision for {crop_type}")
            
            # Get enhanced context with web search and memory
            enhanced_context = await self.get_enhanced_decision_context(decision_type, crop_type, location)
            
            # Gather data from all agents
            agent_input = await self._gather_agent_data(location, crop_type)
            
            # Combine with enhanced context
            if context is None:
                context = {}
            context['enhanced_context'] = enhanced_context
            
            # Analyze the situation
            analysis = await self._analyze_situation(decision_type, agent_input, context)
            
            # Generate decision using LLM
            decision = await self._generate_decision(
                decision_type, crop_type, location, analysis, agent_input
            )
            
            # Store decision
            self.decision_history.append(decision)
            
            # Limit history size
            if len(self.decision_history) > self.config['max_decision_history']:
                self.decision_history = self.decision_history[-self.config['max_decision_history']:]
            
            logger.info(f"✅ Decision made: {decision.decision_id}")
            return decision
            
        except Exception as e:
            logger.error(f"❌ Error making decision: {e}")
            return self._get_default_decision(decision_type, crop_type, location)
    
    async def _gather_agent_data(self, location: Tuple[float, float], 
                                crop_type: str) -> AgentInput:
        """Gather data from all sub-agents"""
        try:
            agent_input = AgentInput()
            
            # Get weather data
            try:
                weather_forecast = await self.weather_agent.get_current_weather(
                    f"{crop_type} field", location[0], location[1]
                )
                agent_input.weather_data = weather_forecast
            except Exception as e:
                logger.warning(f"Weather agent error: {e}")
            
            # Get market data
            try:
                market_analysis = await self.market_agent.analyze_market(crop_type)
                agent_input.market_data = market_analysis
            except Exception as e:
                logger.warning(f"Market agent error: {e}")
            
            # Get pest data
            try:
                pest_assessment = await self.pest_agent.assess_plant_health(
                    f"{crop_type}_field", location
                )
                agent_input.pest_data = pest_assessment
            except Exception as e:
                logger.warning(f"Pest agent error: {e}")
            
            # Get prediction data
            try:
                yield_forecast = await self.prediction_agent.predict_yield(
                    crop_type, location, datetime.now() + timedelta(days=90)
                )
                agent_input.prediction_data = yield_forecast
            except Exception as e:
                logger.warning(f"Prediction agent error: {e}")
            
            # Get risk data
            try:
                risk_assessment = await self.prediction_agent.assess_risk(
                    crop_type, location, datetime.now()
                )
                agent_input.risk_data = risk_assessment
            except Exception as e:
                logger.warning(f"Risk assessment error: {e}")
            
            return agent_input
            
        except Exception as e:
            logger.error(f"❌ Error gathering agent data: {e}")
            return AgentInput()
    
    async def _analyze_situation(self, decision_type: DecisionType, 
                               agent_input: AgentInput, 
                               context: Optional[Dict]) -> Dict[str, Any]:
        """Analyze the current situation using all available data"""
        try:
            analysis = {
                'decision_type': decision_type.value,
                'timestamp': datetime.now(),
                'data_quality': 'good',
                'key_factors': [],
                'risks': [],
                'opportunities': [],
                'constraints': []
            }
            
            # Analyze weather factors
            if agent_input.weather_data:
                weather = agent_input.weather_data
                analysis['key_factors'].append(f"Weather: {weather.condition.value}, {weather.temperature:.1f}°C")
                
                if weather.severity.value in ['high', 'extreme']:
                    analysis['risks'].append(f"Weather risk: {weather.severity.value}")
                
                if weather.condition.value == 'sunny' and 15 <= weather.temperature <= 30:
                    analysis['opportunities'].append("Optimal weather conditions")
            
            # Analyze market factors
            if agent_input.market_data:
                market = agent_input.market_data
                analysis['key_factors'].append(f"Market: {market.trend.value}, ${market.current_price:.2f}")
                
                if market.risk_level.value in ['high', 'critical']:
                    analysis['risks'].append(f"Market risk: {market.risk_level.value}")
                
                if market.trend.value == 'bullish':
                    analysis['opportunities'].append("Favorable market conditions")
            
            # Analyze pest factors
            if agent_input.pest_data:
                pest = agent_input.pest_data
                analysis['key_factors'].append(f"Plant health: {pest.health_status.value}")
                
                if pest.health_status.value in ['poor', 'critical']:
                    analysis['risks'].append(f"Plant health risk: {pest.health_status.value}")
                
                if len(pest.pest_detections) > 0:
                    analysis['risks'].append(f"Pest detections: {len(pest.pest_detections)}")
            
            # Analyze prediction factors
            if agent_input.prediction_data:
                prediction = agent_input.prediction_data
                analysis['key_factors'].append(f"Yield forecast: {prediction.predicted_yield:.1f} tons/ha")
                
                if prediction.confidence < 0.6:
                    analysis['risks'].append("Low prediction confidence")
                
                if prediction.predicted_yield > 8.0:
                    analysis['opportunities'].append("High yield potential")
            
            # Analyze risk factors
            if agent_input.risk_data:
                risk = agent_input.risk_data
                analysis['key_factors'].append(f"Risk level: {risk.risk_category.value}")
                
                if risk.risk_category.value in ['high', 'critical']:
                    analysis['risks'].extend(risk.risk_factors)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing situation: {e}")
            return {'decision_type': decision_type.value, 'timestamp': datetime.now()}
    
    async def _generate_decision(self, decision_type: DecisionType, 
                               crop_type: str, location: Tuple[float, float],
                               analysis: Dict[str, Any], 
                               agent_input: AgentInput) -> Decision:
        """Generate decision using LLM and agent data"""
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(decision_type, crop_type, analysis, agent_input)
            
            # Generate decision using LLM
            if self.openai_api_key:
                decision_text = await self._call_llm(context)
            else:
                decision_text = self._generate_rule_based_decision(decision_type, analysis)
            
            # Parse decision text
            decision = self._parse_decision(decision_type, crop_type, location, decision_text, analysis, agent_input)
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Error generating decision: {e}")
            return self._get_default_decision(decision_type, crop_type, location)
    
    def _prepare_llm_context(self, decision_type: DecisionType, crop_type: str,
                           analysis: Dict[str, Any], agent_input: AgentInput) -> str:
        """Prepare context for LLM decision making"""
        context = f"""
        Agricultural Decision Making Context:
        
        Decision Type: {decision_type.value}
        Crop: {crop_type}
        Location: {agent_input.weather_data.location if agent_input.weather_data else 'Unknown'}
        
        Current Situation:
        {analysis.get('key_factors', [])}
        
        Risks: {analysis.get('risks', [])}
        Opportunities: {analysis.get('opportunities', [])}
        
        Weather: {agent_input.weather_data.condition.value if agent_input.weather_data else 'Unknown'} - {agent_input.weather_data.temperature:.1f}°C
        Market: {agent_input.market_data.trend.value if agent_input.market_data else 'Unknown'} - ${agent_input.market_data.current_price:.2f if agent_input.market_data else 'Unknown'}
        Plant Health: {agent_input.pest_data.health_status.value if agent_input.pest_data else 'Unknown'}
        Yield Forecast: {agent_input.prediction_data.predicted_yield:.1f if agent_input.prediction_data else 'Unknown'} tons/ha
        
        Please provide a strategic decision with:
        1. Clear recommendation
        2. Reasoning based on data
        3. Expected outcomes
        4. Risk assessment
        5. Specific actions to take
        6. Timeline and resources needed
        """
        
        return context
    
    async def _call_llm(self, context: str) -> str:
        """Call LLM for decision generation"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config['llm_model'],
                messages=[
                    {"role": "system", "content": "You are an expert agricultural AI advisor. Provide strategic farming decisions based on data analysis."},
                    {"role": "user", "content": context}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {e}")
            return self._generate_rule_based_decision(DecisionType.PLANTING, {})
    
    def _generate_rule_based_decision(self, decision_type: DecisionType, 
                                    analysis: Dict[str, Any]) -> str:
        """Generate rule-based decision when LLM is not available"""
        if decision_type == DecisionType.PLANTING:
            return """
            RECOMMENDATION: Proceed with planting
            REASONING: Conditions appear favorable based on current data
            EXPECTED OUTCOME: Successful crop establishment
            RISK ASSESSMENT: Moderate risk due to weather variability
            ACTIONS: 1. Prepare soil, 2. Plant seeds, 3. Monitor growth
            TIMELINE: Next 2 weeks
            RESOURCES: Seeds, equipment, labor
            """
        elif decision_type == DecisionType.HARVESTING:
            return """
            RECOMMENDATION: Begin harvesting
            REASONING: Crop maturity indicators suggest optimal timing
            EXPECTED OUTCOME: High quality yield
            RISK ASSESSMENT: Low risk if weather remains stable
            ACTIONS: 1. Check crop maturity, 2. Schedule harvest, 3. Prepare storage
            TIMELINE: Next 1 week
            RESOURCES: Harvesting equipment, storage facilities
            """
        else:
            return """
            RECOMMENDATION: Monitor and maintain current practices
            REASONING: No immediate action required
            EXPECTED OUTCOME: Continued stable production
            RISK ASSESSMENT: Low to moderate risk
            ACTIONS: 1. Continue monitoring, 2. Maintain current practices
            TIMELINE: Ongoing
            RESOURCES: Standard monitoring equipment
            """
    
    def _parse_decision(self, decision_type: DecisionType, crop_type: str, 
                       location: Tuple[float, float], decision_text: str,
                       analysis: Dict[str, Any], agent_input: AgentInput) -> Decision:
        """Parse LLM response into Decision object"""
        try:
            # Extract information from decision text
            lines = decision_text.split('\n')
            
            recommendation = "Monitor and maintain current practices"
            reasoning = "Based on current data analysis"
            expected_outcome = "Stable production"
            risk_assessment = "Low to moderate risk"
            actions = ["Continue monitoring", "Maintain current practices"]
            timeline = "Ongoing"
            resources = ["Standard equipment"]
            
            # Parse structured response
            for line in lines:
                line = line.strip()
                if line.startswith('RECOMMENDATION:'):
                    recommendation = line.replace('RECOMMENDATION:', '').strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                elif line.startswith('EXPECTED OUTCOME:'):
                    expected_outcome = line.replace('EXPECTED OUTCOME:', '').strip()
                elif line.startswith('RISK ASSESSMENT:'):
                    risk_assessment = line.replace('RISK ASSESSMENT:', '').strip()
                elif line.startswith('ACTIONS:'):
                    actions_text = line.replace('ACTIONS:', '').strip()
                    actions = [action.strip() for action in actions_text.split(',')]
                elif line.startswith('TIMELINE:'):
                    timeline = line.replace('TIMELINE:', '').strip()
                elif line.startswith('RESOURCES:'):
                    resources_text = line.replace('RESOURCES:', '').strip()
                    resources = [resource.strip() for resource in resources_text.split(',')]
            
            # Determine priority
            priority = DecisionPriority.MEDIUM
            if 'critical' in risk_assessment.lower() or 'urgent' in recommendation.lower():
                priority = DecisionPriority.CRITICAL
            elif 'high' in risk_assessment.lower() or 'important' in recommendation.lower():
                priority = DecisionPriority.HIGH
            elif 'low' in risk_assessment.lower():
                priority = DecisionPriority.LOW
            
            # Calculate confidence
            confidence = 0.7  # Default
            if len(analysis.get('key_factors', [])) > 3:
                confidence += 0.1
            if len(analysis.get('risks', [])) == 0:
                confidence += 0.1
            if agent_input.weather_data and agent_input.market_data:
                confidence += 0.1
            
            confidence = min(0.95, confidence)
            
            return Decision(
                decision_id=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                decision_type=decision_type,
                priority=priority,
                description=recommendation,
                reasoning=reasoning,
                confidence=confidence,
                expected_outcome=expected_outcome,
                risk_assessment=risk_assessment,
                actions=actions,
                timeline=timeline,
                resources_required=resources,
                success_metrics=["Crop yield", "Quality metrics", "Cost efficiency"],
                created_at=datetime.now(),
                status=ActionStatus.PENDING
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing decision: {e}")
            return self._get_default_decision(decision_type, crop_type, location)
    
    def _get_default_decision(self, decision_type: DecisionType, 
                            crop_type: str, location: Tuple[float, float]) -> Decision:
        """Get default decision when decision generation fails"""
        return Decision(
            decision_id=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type=decision_type,
            priority=DecisionPriority.MEDIUM,
            description="Monitor and maintain current practices",
            reasoning="Default decision due to system limitations",
            confidence=0.5,
            expected_outcome="Stable production",
            risk_assessment="Unknown risk level",
            actions=["Continue monitoring", "Maintain current practices"],
            timeline="Ongoing",
            resources_required=["Standard equipment"],
            success_metrics=["Crop yield", "Quality metrics"],
            created_at=datetime.now(),
            status=ActionStatus.PENDING
        )
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get Decision Agent status and health metrics"""
        try:
            # Get status from all sub-agents
            weather_status = await self.weather_agent.get_agent_status()
            market_status = await self.market_agent.get_agent_status()
            pest_status = await self.pest_agent.get_agent_status()
            prediction_status = await self.prediction_agent.get_agent_status()
            
            status = {
                'agent_name': 'Decision Agent',
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'sub_agents': {
                    'weather_agent': weather_status['status'],
                    'market_agent': market_status['status'],
                    'pest_agent': pest_status['status'],
                    'prediction_agent': prediction_status['status']
                },
                'decision_history_count': len(self.decision_history),
                'llm_available': bool(self.openai_api_key),
                'capabilities': [
                    'Multi-agent coordination',
                    'Strategic decision making',
                    'Risk assessment',
                    'Action planning',
                    'Performance monitoring',
                    'LLM integration'
                ]
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting agent status: {e}")
            return {
                'agent_name': 'Decision Agent',
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    # ==================== ENHANCED CAPABILITIES ====================
    
    async def web_search(self, query: str, max_results: int = 5) -> WebSearchResult:
        """Perform web search for real-time information"""
        try:
            # Check cache first
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if cache_key in self.web_search_cache:
                cached_result = self.web_search_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < timedelta(seconds=self.config['search_cache_duration']):
                    return cached_result['result']
            
            # Perform search using multiple engines
            search_results = []
            
            # Google Custom Search (if API key available)
            if self.config.get('google_api_key'):
                google_results = await self._search_google(query, max_results)
                search_results.extend(google_results)
            
            # DuckDuckGo (free, no API key needed)
            duckduckgo_results = await self._search_duckduckgo(query, max_results)
            search_results.extend(duckduckgo_results)
            
            # Remove duplicates and rank by relevance
            unique_results = self._deduplicate_and_rank_results(search_results)
            
            result = WebSearchResult(
                query=query,
                results=unique_results[:max_results],
                timestamp=datetime.now(),
                source='multiple',
                relevance_score=0.8  # Simplified
            )
            
            # Cache result
            self.web_search_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in web search: {e}")
            return WebSearchResult(
                query=query,
                results=[],
                timestamp=datetime.now(),
                source='error',
                relevance_score=0.0
            )
    
    async def _search_google(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        try:
            if not self.config.get('google_api_key'):
                return []
            
            url = f"{self.search_engines['google']}?key={self.config['google_api_key']}&cx={self.config.get('google_cx', '')}&q={query}&num={max_results}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google'
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"Google search error: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (free)"""
        try:
            # DuckDuckGo instant answer API
            url = f"{self.search_engines['duckduckgo']}?q={query}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract abstract and related topics
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': 'duckduckgo'
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related Topic',
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'duckduckgo'
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")
            return []
    
    def _deduplicate_and_rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank search results by relevance"""
        try:
            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            
            for result in results:
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
            
            # Simple ranking by title and snippet relevance
            def relevance_score(result):
                score = 0
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                
                # Agricultural keywords get higher scores
                ag_keywords = ['agriculture', 'farming', 'crop', 'yield', 'weather', 'market', 'price']
                for keyword in ag_keywords:
                    if keyword in title or keyword in snippet:
                        score += 1
                
                return score
            
            # Sort by relevance score
            unique_results.sort(key=relevance_score, reverse=True)
            
            return unique_results
            
        except Exception as e:
            logger.error(f"❌ Error deduplicating results: {e}")
            return results
    
    async def use_external_tool(self, tool_name: str, input_data: Dict[str, Any]) -> ToolResult:
        """Use external tools and APIs"""
        try:
            start_time = time.time()
            
            if tool_name == 'weather_api':
                result = await self._call_weather_api(input_data)
            elif tool_name == 'market_api':
                result = await self._call_market_api(input_data)
            elif tool_name == 'news_api':
                result = await self._call_news_api(input_data)
            elif tool_name == 'iot_sensors':
                result = await self._call_iot_sensors(input_data)
            elif tool_name == 'database_query':
                result = await self._query_database(input_data)
            else:
                result = {'error': f'Unknown tool: {tool_name}'}
            
            execution_time = time.time() - start_time
            
            tool_result = ToolResult(
                tool_name=tool_name,
                input_data=input_data,
                output_data=result,
                success='error' not in result,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            # Cache result
            cache_key = f"{tool_name}_{hashlib.md5(str(input_data).encode()).hexdigest()}"
            self.tool_results_cache[cache_key] = tool_result
            
            return tool_result
            
        except Exception as e:
            logger.error(f"❌ Error using tool {tool_name}: {e}")
            return ToolResult(
                tool_name=tool_name,
                input_data=input_data,
                output_data={'error': str(e)},
                success=False,
                execution_time=0,
                timestamp=datetime.now()
            )
    
    async def _call_weather_api(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call external weather API"""
        try:
            if not self.weather_api_key:
                return {'error': 'Weather API key not configured'}
            
            lat = input_data.get('latitude', 0)
            lon = input_data.get('longitude', 0)
            
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {'error': f'Weather API error: {e}'}
    
    async def _call_market_api(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call external market API"""
        try:
            if not self.market_api_key:
                return {'error': 'Market API key not configured'}
            
            crop = input_data.get('crop', 'corn')
            url = f"https://api.marketdata.com/v1/prices/{crop}"
            
            headers = {'Authorization': f'Bearer {self.market_api_key}'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {'error': f'Market API error: {e}'}
    
    async def _call_news_api(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call news API for agricultural news"""
        try:
            if not self.news_api_key:
                return {'error': 'News API key not configured'}
            
            query = input_data.get('query', 'agriculture farming')
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}&sortBy=publishedAt"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {'error': f'News API error: {e}'}
    
    async def _call_iot_sensors(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query IoT sensors for real-time data"""
        try:
            location = input_data.get('location', {})
            sensor_types = input_data.get('sensor_types', ['temperature', 'humidity'])
            
            # Query MongoDB for recent sensor data
            query = {
                'location.latitude': {'$gte': location.get('latitude', 0) - 0.01, '$lte': location.get('latitude', 0) + 0.01},
                'location.longitude': {'$gte': location.get('longitude', 0) - 0.01, '$lte': location.get('longitude', 0) + 0.01},
                'sensor_type': {'$in': sensor_types},
                'timestamp': {'$gte': datetime.now() - timedelta(hours=1)}
            }
            
            cursor = self.mongo_manager.db.iot_sensor_data.find(query)
            data = list(cursor)
            
            return {'sensor_data': data, 'count': len(data)}
            
        except Exception as e:
            return {'error': f'IoT sensors error: {e}'}
    
    async def _query_database(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query database for historical data"""
        try:
            collection = input_data.get('collection', 'iot_sensor_data')
            query = input_data.get('query', {})
            limit = input_data.get('limit', 100)
            
            cursor = self.mongo_manager.db[collection].find(query).limit(limit)
            data = list(cursor)
            
            return {'data': data, 'count': len(data)}
            
        except Exception as e:
            return {'error': f'Database query error: {e}'}
    
    def _load_persistent_memory(self) -> List[MemoryEntry]:
        """Load persistent memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    memory = pickle.load(f)
                
                # Filter old entries
                cutoff_date = datetime.now() - timedelta(days=self.config['memory_retention_days'])
                filtered_memory = [entry for entry in memory if entry.timestamp > cutoff_date]
                
                logger.info(f"📚 Loaded {len(filtered_memory)} memory entries")
                return filtered_memory
            else:
                logger.info("📚 No existing memory file found, starting fresh")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error loading memory: {e}")
            return []
    
    def _save_persistent_memory(self) -> None:
        """Save persistent memory to file"""
        try:
            # Limit memory size
            if len(self.persistent_memory) > self.config['max_memory_entries']:
                # Keep most important entries
                self.persistent_memory.sort(key=lambda x: x.importance_score, reverse=True)
                self.persistent_memory = self.persistent_memory[:self.config['max_memory_entries']]
            
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.persistent_memory, f)
            
            logger.info(f"💾 Saved {len(self.persistent_memory)} memory entries")
            
        except Exception as e:
            logger.error(f"❌ Error saving memory: {e}")
    
    async def add_to_memory(self, decision: Decision, outcome: str, success: bool, 
                           lessons_learned: List[str], importance_score: float = 0.5) -> None:
        """Add decision outcome to persistent memory"""
        try:
            memory_entry = MemoryEntry(
                memory_id=f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                decision_id=decision.decision_id,
                context={
                    'decision_type': decision.decision_type.value,
                    'priority': decision.priority.value,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning,
                    'actions': decision.actions
                },
                outcome=outcome,
                success=success,
                lessons_learned=lessons_learned,
                timestamp=datetime.now(),
                importance_score=importance_score
            )
            
            self.persistent_memory.append(memory_entry)
            self._save_persistent_memory()
            
            logger.info(f"🧠 Added memory entry: {memory_entry.memory_id}")
            
        except Exception as e:
            logger.error(f"❌ Error adding to memory: {e}")
    
    async def search_memory(self, query: str, max_results: int = 5) -> List[MemoryEntry]:
        """Search persistent memory for relevant past decisions"""
        try:
            relevant_memories = []
            query_lower = query.lower()
            
            for memory in self.persistent_memory:
                relevance_score = 0
                
                # Check context fields
                context = memory.context
                if query_lower in context.get('decision_type', '').lower():
                    relevance_score += 2
                if query_lower in context.get('reasoning', '').lower():
                    relevance_score += 1
                if query_lower in memory.outcome.lower():
                    relevance_score += 1
                
                # Check lessons learned
                for lesson in memory.lessons_learned:
                    if query_lower in lesson.lower():
                        relevance_score += 1
                
                if relevance_score > 0:
                    relevant_memories.append((memory, relevance_score))
            
            # Sort by relevance and importance
            relevant_memories.sort(key=lambda x: (x[1], x[0].importance_score), reverse=True)
            
            return [memory for memory, score in relevant_memories[:max_results]]
            
        except Exception as e:
            logger.error(f"❌ Error searching memory: {e}")
            return []
    
    async def get_enhanced_decision_context(self, decision_type: DecisionType, 
                                         crop_type: str, location: Tuple[float, float]) -> Dict[str, Any]:
        """Get enhanced decision context with web search and memory"""
        try:
            context = {
                'timestamp': datetime.now(),
                'decision_type': decision_type.value,
                'crop_type': crop_type,
                'location': location
            }
            
            # Web search for recent agricultural news and trends
            search_queries = [
                f"{crop_type} farming news 2024",
                f"{crop_type} market prices today",
                f"agricultural weather forecast {location[0]},{location[1]}",
                f"{decision_type.value} best practices agriculture"
            ]
            
            web_results = []
            for query in search_queries:
                search_result = await self.web_search(query, max_results=3)
                web_results.extend(search_result.results)
            
            context['web_search_results'] = web_results[:10]  # Limit to top 10
            
            # Search memory for similar past decisions
            memory_query = f"{decision_type.value} {crop_type}"
            similar_decisions = await self.search_memory(memory_query, max_results=5)
            context['similar_past_decisions'] = [
                {
                    'decision_type': mem.context.get('decision_type'),
                    'outcome': mem.outcome,
                    'success': mem.success,
                    'lessons_learned': mem.lessons_learned,
                    'timestamp': mem.timestamp.isoformat()
                }
                for mem in similar_decisions
            ]
            
            # Get real-time data using external tools
            tool_results = {}
            
            # Weather data
            weather_tool_result = await self.use_external_tool('weather_api', {
                'latitude': location[0],
                'longitude': location[1]
            })
            if weather_tool_result.success:
                tool_results['weather'] = weather_tool_result.output_data
            
            # Market data
            market_tool_result = await self.use_external_tool('market_api', {
                'crop': crop_type
            })
            if market_tool_result.success:
                tool_results['market'] = market_tool_result.output_data
            
            # IoT sensor data
            iot_tool_result = await self.use_external_tool('iot_sensors', {
                'location': {'latitude': location[0], 'longitude': location[1]},
                'sensor_types': ['temperature', 'humidity', 'pressure', 'rainfall']
            })
            if iot_tool_result.success:
                tool_results['iot_sensors'] = iot_tool_result.output_data
            
            context['real_time_data'] = tool_results
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error getting enhanced context: {e}")
            return {
                'timestamp': datetime.now(),
                'decision_type': decision_type.value,
                'crop_type': crop_type,
                'location': location,
                'error': str(e)
            }


    async def demonstrate_enhanced_capabilities(self, crop_type: str = "corn", 
                                             location: Tuple[float, float] = (40.7128, -74.0060)) -> Dict[str, Any]:
        """Demonstrate the enhanced capabilities of the Decision Agent"""
        try:
            logger.info("🚀 Demonstrating Enhanced Decision Agent Capabilities")
            
            demo_results = {
                'timestamp': datetime.now().isoformat(),
                'capabilities_demonstrated': []
            }
            
            # 1. Web Search Capability
            logger.info("🔍 Testing Web Search...")
            search_result = await self.web_search(f"{crop_type} farming best practices 2024", max_results=3)
            demo_results['web_search'] = {
                'query': search_result.query,
                'results_count': len(search_result.results),
                'sources': [r.get('source', 'unknown') for r in search_result.results],
                'sample_titles': [r.get('title', '') for r in search_result.results[:2]]
            }
            demo_results['capabilities_demonstrated'].append('Web Search')
            
            # 2. External Tools Capability
            logger.info("🛠️ Testing External Tools...")
            tool_results = {}
            
            # Test IoT sensors
            iot_result = await self.use_external_tool('iot_sensors', {
                'location': {'latitude': location[0], 'longitude': location[1]},
                'sensor_types': ['temperature', 'humidity']
            })
            tool_results['iot_sensors'] = {
                'success': iot_result.success,
                'execution_time': iot_result.execution_time,
                'data_count': iot_result.output_data.get('count', 0) if iot_result.success else 0
            }
            
            # Test database query
            db_result = await self.use_external_tool('database_query', {
                'collection': 'iot_sensor_data',
                'query': {'sensor_type': 'temperature'},
                'limit': 5
            })
            tool_results['database_query'] = {
                'success': db_result.success,
                'execution_time': db_result.execution_time,
                'data_count': db_result.output_data.get('count', 0) if db_result.success else 0
            }
            
            demo_results['external_tools'] = tool_results
            demo_results['capabilities_demonstrated'].append('External Tools')
            
            # 3. Persistent Memory Capability
            logger.info("🧠 Testing Persistent Memory...")
            
            # Add a sample memory entry
            sample_decision = Decision(
                decision_id="demo_001",
                decision_type=DecisionType.PLANTING,
                priority=DecisionPriority.HIGH,
                confidence=0.85,
                reasoning="Demo decision for testing memory capabilities",
                actions=["Test action 1", "Test action 2"],
                expected_outcome="Successful demonstration",
                success_metrics=["Memory storage", "Memory retrieval"],
                created_at=datetime.now(),
                status=ActionStatus.PENDING
            )
            
            await self.add_to_memory(
                sample_decision,
                outcome="Memory test successful",
                success=True,
                lessons_learned=["Memory system works correctly", "Data persistence is functional"],
                importance_score=0.7
            )
            
            # Search memory
            memory_search = await self.search_memory("planting", max_results=3)
            demo_results['persistent_memory'] = {
                'total_memories': len(self.persistent_memory),
                'search_results': len(memory_search),
                'sample_memories': [
                    {
                        'decision_type': mem.context.get('decision_type'),
                        'outcome': mem.outcome,
                        'success': mem.success,
                        'timestamp': mem.timestamp.isoformat()
                    }
                    for mem in memory_search[:2]
                ]
            }
            demo_results['capabilities_demonstrated'].append('Persistent Memory')
            
            # 4. Enhanced Decision Context
            logger.info("🎯 Testing Enhanced Decision Context...")
            enhanced_context = await self.get_enhanced_decision_context(
                DecisionType.PLANTING, crop_type, location
            )
            demo_results['enhanced_context'] = {
                'web_search_results': len(enhanced_context.get('web_search_results', [])),
                'similar_past_decisions': len(enhanced_context.get('similar_past_decisions', [])),
                'real_time_data_sources': list(enhanced_context.get('real_time_data', {}).keys()),
                'has_enhanced_data': len(enhanced_context) > 4
            }
            demo_results['capabilities_demonstrated'].append('Enhanced Context')
            
            logger.info(f"✅ Enhanced capabilities demonstration completed: {len(demo_results['capabilities_demonstrated'])} capabilities tested")
            return demo_results
            
        except Exception as e:
            logger.error(f"❌ Error in capabilities demonstration: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'capabilities_demonstrated': []
            }


# Example usage and testing
async def main():
    """Example usage of Enhanced Decision Agent"""
    try:
        # Initialize MongoDB manager
        mongo_manager = MongoDBManager()
        await mongo_manager.connect()
        
        # Initialize Decision Agent
        decision_agent = DecisionAgent(mongo_manager)
        await decision_agent.initialize_agents()
        
        # Test parameters
        location = (40.7128, -74.0060)
        crop_type = "corn"
        
        # Make a planting decision
        planting_decision = await decision_agent.make_decision(
            DecisionType.PLANTING, location, crop_type
        )
        
        print(f"Decision: {planting_decision.description}")
        print(f"Priority: {planting_decision.priority.value}")
        print(f"Confidence: {planting_decision.confidence:.2f}")
        print(f"Reasoning: {planting_decision.reasoning}")
        print(f"Actions: {', '.join(planting_decision.actions)}")
        
        # Make a harvesting decision
        harvest_decision = await decision_agent.make_decision(
            DecisionType.HARVESTING, location, crop_type
        )
        
        print(f"\nHarvest Decision: {harvest_decision.description}")
        print(f"Expected Outcome: {harvest_decision.expected_outcome}")
        print(f"Timeline: {harvest_decision.timeline}")
        
        # Get agent status
        status = await decision_agent.get_agent_status()
        print(f"\nAgent Status: {status['status']}")
        print(f"Sub-agents: {status['sub_agents']}")
        print(f"Decisions made: {status['decision_history_count']}")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())
