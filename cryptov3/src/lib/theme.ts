export const theme = {
  colors: {
    primary: '#7e22ce', // Purple 800
    secondary: '#f5f3ff', // Purple 50
    accent: '#a855f7', // Purple 500
    background: '#ffffff',
    darkPurple: '#581c87', // Purple 900
    mediumPurple: '#9333ea', // Purple 600
    lightPurple: '#c084fc', // Purple 400
    veryLightPurple: '#f3e8ff', // Purple 100
  },
};

export type TimeframeOption = '15m' | '1h' | '4h' | '1d' | '7d';

export const timeframeOptions: TimeframeOption[] = ['15m', '1h', '4h', '1d', '7d'];
