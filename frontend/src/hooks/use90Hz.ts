import { useCallback, useRef, useEffect } from 'react';
import { useSpring, useTransform, useMotionValue, useAnimation } from 'framer-motion';

// 90Hz optimized hook for smooth interactions
export const use90Hz = () => {
  const controls = useAnimation();
  const motionValue = useMotionValue(0);
  
  // Optimized spring configuration for 90Hz
  const springConfig = {
    type: "spring" as const,
    stiffness: 400,
    damping: 30,
    mass: 0.8,
  };

  // Smooth hover animation
  const hoverAnimation = useCallback((scale: number = 1.02) => {
    controls.start({
      scale,
      transition: springConfig
    });
  }, [controls, springConfig]);

  // Quick tap animation
  const tapAnimation = useCallback((scale: number = 0.98) => {
    controls.start({
      scale,
      transition: {
        duration: 0.05, // ~4.5 frames at 90Hz
        ease: [0.4, 0, 0.2, 1]
      }
    });
  }, [controls]);

  // Reset animation
  const resetAnimation = useCallback(() => {
    controls.start({
      scale: 1,
      transition: springConfig
    });
  }, [controls, springConfig]);

  return {
    controls,
    motionValue,
    hoverAnimation,
    tapAnimation,
    resetAnimation,
    springConfig
  };
};

// Performance-optimized scroll hook
export const use90HzScroll = () => {
  const scrollY = useMotionValue(0);
  const scrollX = useMotionValue(0);
  
  // Parallax transforms optimized for 90Hz
  const y = useTransform(scrollY, [0, 1000], [0, -100]);
  const x = useTransform(scrollX, [0, 1000], [0, -50]);
  
  // Smooth scroll with 90Hz optimization
  const smoothScroll = useCallback((targetY: number) => {
    scrollY.set(targetY, {
      type: "spring",
      stiffness: 300,
      damping: 30,
      mass: 1
    });
  }, [scrollY]);

  return {
    scrollY,
    scrollX,
    y,
    x,
    smoothScroll
  };
};

// 90Hz optimized stagger hook
export const use90HzStagger = (itemCount: number) => {
  const staggerDelay = 0.05; // ~4.5 frames at 90Hz
  
  const getStaggerDelay = useCallback((index: number) => {
    return index * staggerDelay;
  }, [staggerDelay]);

  const getStaggerVariants = useCallback(() => ({
    initial: { 
      opacity: 0, 
      y: 20,
      scale: 0.95
    },
    animate: (index: number) => ({
      opacity: 1, 
      y: 0,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 350,
        damping: 25,
        mass: 0.7,
        delay: getStaggerDelay(index)
      }
    })
  }), [getStaggerDelay]);

  return {
    getStaggerDelay,
    getStaggerVariants,
    staggerDelay
  };
};

// Performance monitoring hook
export const use90HzPerformance = () => {
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  const fps = useRef(0);

  useEffect(() => {
    const measureFPS = () => {
      frameCount.current++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime.current >= 1000) {
        fps.current = Math.round((frameCount.current * 1000) / (currentTime - lastTime.current));
        frameCount.current = 0;
        lastTime.current = currentTime;
        
        // Log performance warnings
        if (fps.current < 60) {
          console.warn(`Low FPS detected: ${fps.current}fps`);
        }
      }
      
      requestAnimationFrame(measureFPS);
    };

    requestAnimationFrame(measureFPS);
  }, []);

  return {
    fps: fps.current,
    is90Hz: fps.current >= 85, // Consider 85+ as 90Hz capable
    is60Hz: fps.current >= 55 && fps.current < 85,
    isLowFPS: fps.current < 55
  };
};
