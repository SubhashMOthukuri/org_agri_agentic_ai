import React, { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

interface WeatherChartProps {
  data: any;
}

export const WeatherChart: React.FC<WeatherChartProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data?.forecast || data.forecast.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = 800;
    const height = 300;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    // Clear previous content
    svg.selectAll('*').remove();

    // Create scales with real data
    const xScale = d3.scaleTime()
      .domain(d3.extent(data.forecast, (d: any) => new Date(d.date)))
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data.forecast, (d: any) => d.temperature))
      .range([height - margin.bottom, margin.top]);

    // Create line generator
    const line = d3.line<any>()
      .x(d => xScale(new Date(d.date)))
      .y(d => yScale(d.temperature))
      .curve(d3.curveMonotoneX);

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%m/%d')));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).tickFormat(d => `${d}°C`));

    // Add line
    svg.append('path')
      .datum(data.forecast)
      .attr('fill', 'none')
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add dots
    svg.selectAll('.dot')
      .data(data.forecast)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(new Date(d.date)))
      .attr('cy', d => yScale(d.temperature))
      .attr('r', 4)
      .attr('fill', '#22c55e')
      .attr('opacity', 0.8);

    // Add precipitation bars if data exists
    if (data.forecast.some((d: any) => d.precipitation_probability !== undefined)) {
      const precipitationScale = d3.scaleLinear()
        .domain([0, d3.max(data.forecast, (d: any) => d.precipitation_probability) || 1])
        .range([0, 20]);

      svg.selectAll('.precipitation')
        .data(data.forecast)
        .enter()
        .append('rect')
        .attr('class', 'precipitation')
        .attr('x', d => xScale(new Date(d.date)) - 5)
        .attr('y', height - margin.bottom - precipitationScale(d.precipitation_probability))
        .attr('width', 10)
        .attr('height', d => precipitationScale(d.precipitation_probability))
        .attr('fill', '#3b82f6')
        .attr('opacity', 0.6);
    }

  }, [data]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-900">7-Day Temperature Forecast</h4>
        <p className="text-xs text-gray-500">Blue bars indicate precipitation probability</p>
      </div>
      <svg ref={svgRef} width="100%" height="300" />
    </motion.div>
  );
};
