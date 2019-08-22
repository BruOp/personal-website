import React from "react";
import Layout from "../components/layout";

const AboutMe = () => {
  return (
    <Layout>
      <h1>Hello there!</h1>
      <p>
        I'm Bruno Opsenica. I've recently decided to try and make my hobby of learning about 3D programming and computer
        graphics into a career. In order to do that, I've created this blog as a way of chronicling the things I'm
        learning about. It's meant to be a mechanism to make my learning visible to others, get feedback and
        (eventually, hopefully) provide help to others if they decide to go down the same path.
      </p>
      <p>
        I live in Toronto, and work as a software engineer at a startup called <a href="https://delphia.com">Delphia</a>
        . Before that, I worked at the direct precursor of Delphia, <a href="https://voxpoplabs.com">Vox Pop Labs</a>,
        where I got an opportunity to build a few different interactive data visualizations, like{" "}
        <a href="http://votecompass.com">Vote Compass</a> and <a href="https://echoes.cbc.ca">Echoes</a>
        <sup>1</sup>.
      </p>

      <p>
        Before that, I attended the University of Toronto where I specialized in Physics and graduated in 2012. It was
        while taking courses on fluid mechanics and geophysics that I discovered my love of programming, and a fourth
        year course in computer graphics sealed my fate.
      </p>

      <p>[1] Not sure how long this link will actually stay up!</p>
    </Layout>
  );
};

export default AboutMe;
