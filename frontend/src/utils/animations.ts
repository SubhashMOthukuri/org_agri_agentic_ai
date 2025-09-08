// 90Hz Optimized Animation Configurations
export const animations = {
  // 90Hz optimized spring configurations
  spring: {
    // For micro-interactions (buttons, cards)
    micro: {
      type: "spring",
      stiffness: 400,
      damping: 30,
      mass: 0.8,
    },
    // For page transitions
    page: {
      type: "spring",
      stiffness: 300,
      damping: 30,
      mass: 1,
    },
    // For modal/overlay animations
    modal: {
      type: "spring",
      stiffness: 500,
      damping: 35,
      mass: 0.6,
    },
    // For list items
    list: {
      type: "spring",
      stiffness: 350,
      damping: 25,
      mass: 0.7,
    }
  },

  // Optimized easing functions for 90Hz
  easing: {
    // For smooth hover effects
    hover: [0.25, 0.46, 0.45, 0.94],
    // For quick interactions
    quick: [0.4, 0, 0.2, 1],
    // For smooth transitions
    smooth: [0.25, 0.1, 0.25, 1],
    // For bouncy effects
    bouncy: [0.68, -0.55, 0.265, 1.55],
  },

  // Duration optimized for 90Hz (multiples of 11.11ms)
  duration: {
    instant: 0.05,    // ~4.5 frames at 90Hz
    fast: 0.1,        // ~9 frames at 90Hz
    normal: 0.2,      // ~18 frames at 90Hz
    slow: 0.4,        // ~36 frames at 90Hz
    slower: 0.6,      // ~54 frames at 90Hz
  },

  // 90Hz optimized stagger delays
  stagger: {
    fast: 0.05,       // ~4.5 frames
    normal: 0.1,      // ~9 frames
    slow: 0.15,       // ~13.5 frames
  }
};

// Performance-optimized motion variants
export const motionVariants = {
  // Fade in with 90Hz optimization
  fadeIn: {
    initial: { opacity: 0 },
    animate: { 
      opacity: 1,
      transition: {
        duration: animations.duration.fast,
        ease: animations.easing.smooth
      }
    },
    exit: { 
      opacity: 0,
      transition: {
        duration: animations.duration.fast,
        ease: animations.easing.quick
      }
    }
  },

  // Slide up with spring physics
  slideUp: {
    initial: { 
      opacity: 0, 
      y: 20,
      scale: 0.95
    },
    animate: { 
      opacity: 1, 
      y: 0,
      scale: 1,
      transition: animations.spring.page
    },
    exit: { 
      opacity: 0, 
      y: -20,
      scale: 0.95,
      transition: {
        duration: animations.duration.fast,
        ease: animations.easing.quick
      }
    }
  },

  // Scale with micro-interaction optimization
  scale: {
    initial: { scale: 1 },
    hover: { 
      scale: 1.02,
      transition: animations.spring.micro
    },
    tap: { 
      scale: 0.98,
      transition: {
        duration: animations.duration.instant,
        ease: animations.easing.quick
      }
    }
  },

  // Card hover with 90Hz optimization
  cardHover: {
    initial: { 
      y: 0,
      boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
    },
    hover: { 
      y: -2,
      boxShadow: "0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
      transition: animations.spring.micro
    }
  },

  // List item stagger animation
  listItem: {
    initial: { 
      opacity: 0, 
      x: -20,
      scale: 0.95
    },
    animate: (index: number) => ({
      opacity: 1, 
      x: 0,
      scale: 1,
      transition: {
        ...animations.spring.list,
        delay: index * animations.stagger.fast
      }
    })
  },

  // Loading spinner optimized for 90Hz
  spinner: {
    animate: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }
    }
  },

  // Pulse animation for live indicators
  pulse: {
    animate: {
      scale: [1, 1.1, 1],
      opacity: [1, 0.7, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: animations.easing.smooth
      }
    }
  }
};

// Performance-optimized hover handlers
export const hoverHandlers = {
  // Card hover with 90Hz optimization
  card: {
    whileHover: { 
      y: -2,
      boxShadow: "0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
      transition: animations.spring.micro
    },
    whileTap: { 
      scale: 0.98,
      transition: {
        duration: animations.duration.instant,
        ease: animations.easing.quick
      }
    }
  },

  // Button hover with micro-interaction
  button: {
    whileHover: { 
      scale: 1.02,
      transition: animations.spring.micro
    },
    whileTap: { 
      scale: 0.98,
      transition: {
        duration: animations.duration.instant,
        ease: animations.easing.quick
      }
    }
  },

  // Icon hover with smooth scaling
  icon: {
    whileHover: { 
      scale: 1.1,
      transition: animations.spring.micro
    },
    whileTap: { 
      scale: 0.9,
      transition: {
        duration: animations.duration.instant,
        ease: animations.easing.quick
      }
    }
  }
};

// 90Hz optimized transition presets
export const transitions = {
  // Page transitions
  page: {
    type: "spring",
    stiffness: 300,
    damping: 30,
    mass: 1,
  },
  
  // Modal transitions
  modal: {
    type: "spring",
    stiffness: 500,
    damping: 35,
    mass: 0.6,
  },
  
  // List animations
  list: {
    type: "spring",
    stiffness: 350,
    damping: 25,
    mass: 0.7,
  },
  
  // Quick interactions
  quick: {
    duration: 0.1,
    ease: [0.4, 0, 0.2, 1]
  },
  
  // Smooth interactions
  smooth: {
    duration: 0.2,
    ease: [0.25, 0.1, 0.25, 1]
  }
};
